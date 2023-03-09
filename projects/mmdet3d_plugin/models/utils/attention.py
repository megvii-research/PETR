# flash-attention 
import math
import torch
import torch.nn as nn
from torch.nn.init import (
    xavier_uniform_,
    constant_,
    xavier_normal_
)
from torch.nn.functional import linear

from einops import rearrange
from mmcv.runner import auto_fp16
from mmcv.runner.base_module import BaseModule

from flash_attn.flash_attn_interface import flash_attn_unpadded_kvpacked_func
from flash_attn.bert_padding import unpad_input, pad_input, index_first_axis


def _in_projection_packed(q, k, v, w, b = None):
    w_q, w_k, w_v = w.chunk(3)
    if b is None:
        b_q = b_k = b_v = None
    else:
        b_q, b_k, b_v = b.chunk(3)
    return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


class FlashAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.1)
    """
    def __init__(self, softmax_scale=None, attention_dropout=0.0, device=None, dtype=None):
        super().__init__()
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout
        self.fp16_enabled = True

    @auto_fp16(apply_to=('q', 'kv'), out_fp32=True)
    def forward(self, q, kv, 
                causal=False, 
                key_padding_mask=None):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q: The tensor containing the query. (B, T, H, D) 
            kv: The tensor containing the key, and value. (B, S, 2, H, D) 
            key_padding_mask: a bool tensor of shape (B, S)
        """
        assert q.dtype in [torch.float16, torch.bfloat16] and kv.dtype in [torch.float16, torch.bfloat16]
        assert q.is_cuda and kv.is_cuda
        assert q.shape[0] == kv.shape[0] and q.shape[-2] == kv.shape[-2] and q.shape[-1] == kv.shape[-1]

        batch_size = q.shape[0]
        seqlen_q, seqlen_k = q.shape[1], kv.shape[1]
        if key_padding_mask is None:
            q, kv = rearrange(q, 'b s ... -> (b s) ...'), rearrange(kv, 'b s ... -> (b s) ...')
            max_sq, max_sk = seqlen_q, seqlen_k 
            cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32,
                                    device=q.device)
            cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32,
                                    device=kv.device)                    
            output = flash_attn_unpadded_kvpacked_func(
                q, kv, cu_seqlens_q, cu_seqlens_k, max_sq, max_sk,
                self.dropout_p if self.training else 0.0,
                softmax_scale=self.softmax_scale, causal=causal
            )
            output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
        else:
            nheads = kv.shape[-2]
            q = rearrange(q, 'b s ... -> (b s) ...')
            max_sq = seqlen_q
            cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32,
                                    device=q.device)
            x = rearrange(kv, 'b s two h d -> b s (two h d)')
            x_unpad, indices, cu_seqlens_k, max_sk = unpad_input(x, key_padding_mask)
            x_unpad = rearrange(x_unpad, 'nnz (two h d) -> nnz two h d', two=2, h=nheads)
            output_unpad = flash_attn_unpadded_kvpacked_func(
                q, x_unpad, cu_seqlens_q, cu_seqlens_k, max_sq, max_sk,
                self.dropout_p if self.training else 0.0,
                softmax_scale=self.softmax_scale, causal=causal
            )
            output = rearrange(output_unpad, '(b s) ... -> b s ...', b=batch_size)

        return output, None


class FlashMHA(nn.Module):

    def __init__(self, embed_dim, num_heads, bias=True, batch_first=True, attention_dropout=0.0,
                 causal=False, device=None, dtype=None, **kwargs) -> None:
        assert batch_first
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal
        self.bias = bias

        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, "Only support head_dim <= 128 and divisible by 8"

        self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dim, embed_dim)))
        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.inner_attn = FlashAttention(attention_dropout=attention_dropout, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        
    def forward(self, q, k, v, key_padding_mask=None):
        """x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim)
        key_padding_mask: bool tensor of shape (batch, seqlen)
        """
        # q, k, v = self.Wq(q), self.Wk(k), self.Wv(v)
        q, k, v = _in_projection_packed(q, k, v, self.in_proj_weight, self.in_proj_bias)
        q = rearrange(q, 'b s (h d) -> b s h d', h=self.num_heads)
        k = rearrange(k, 'b s (h d) -> b s h d', h=self.num_heads)
        v = rearrange(v, 'b s (h d) -> b s h d', h=self.num_heads)
        kv = torch.stack([k, v], dim=2)
        
        context, attn_weights = self.inner_attn(q, kv, key_padding_mask=key_padding_mask, causal=self.causal)
        return self.out_proj(rearrange(context, 'b s h d -> b s (h d)')), attn_weights


# if __name__ == '__main__':
#     batch_size = 64
#     nheads = 16
#     seqlen = 1024
#     n = 1024
#     d = n // nheads
#     dropout_p = 0.1
#     causal = False
#     dtype = torch.float16
#     device = 'cuda'

#     q = torch.randn(batch_size, 100, n, device='cuda', dtype=dtype, requires_grad=True)
#     k = torch.randn(batch_size, seqlen, n, device='cuda', dtype=dtype, requires_grad=True)
#     v = torch.randn(batch_size, seqlen, n, device='cuda', dtype=dtype, requires_grad=True)
#     key_padding_mask = k.new_ones(batch_size, seqlen)
#     model = FlashMHA(n, nheads, device=device, attention_dropout=dropout_p, dtype=dtype)
#     model.eval()
#     model2 = nn.MultiheadAttention(n, nheads, device=device, dtype=dtype)
#     model2.eval()
#     model.Wq.weight.data.fill_(1e-4)
#     model.Wq.bias.data.fill_(0)
#     model.Wk.weight.data.fill_(1e-3)
#     model.Wk.bias.data.fill_(0)
#     model.Wv.weight.data.fill_(1e-2)
#     model.Wv.bias.data.fill_(0)
#     model.out_proj.weight.data.fill_(1e-2)
#     model.out_proj.bias.data.fill_(0)

#     model2.in_proj_weight.data.fill_(1e-3)
#     model2.in_proj_weight.data[:n, :].fill_(1e-4)
#     model2.in_proj_weight.data[-n:, :].fill_(1e-2)
#     model2.in_proj_bias.data.fill_(0)
#     model2.out_proj.weight.data.fill_(1e-2)
#     model2.out_proj.bias.data.fill_(0)
#     out1 = model(q, k, v)[0]
#     out2 = model2(q.transpose(1, 0), k.transpose(1, 0), v.transpose(1, 0))[0].transpose(1, 0)
#     print((out1 - out2).sum(dim=-1).flatten()[:100])
