import torch.nn as nn
import torch.nn.functional as F
import torch
from mmcv.runner import force_fp32, auto_fp16

from .utils import weight_reduce_loss
from ..builder import DISTILL_LOSSES
import math
import numpy as np
import os
import cv2

@DISTILL_LOSSES.register_module()
class ChannelWiseDivergence(nn.Module):

    """PyTorch version of `Channel-wise Distillation for Semantic Segmentation
     <https://arxiv.org/abs/2011.13256>`_.
   
    Args:
        student_channels(int): Number of channels in the student's feature map.
        teacher_channels(int): Number of channels in the teacher's feature map.
        name(str): 
        tau (float, optional): Temperature coefficient. Defaults to 1.0.
        weight (float, optional): Weight of loss.Defaults to 1.0.
        
    """
    def __init__(self,
                 student_channels,
                 teacher_channels,
                 name,
                 tau=1.0,
                 weight=1.0,
                 ):
        super(ChannelWiseDivergence, self).__init__()
        self.tau = tau
        self.loss_weight = weight
    
        if student_channels != teacher_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.align = None
        
        

    # @force_fp32(apply_to=('preds_S'))
    def forward(self,
                preds_S,
                preds_T):
        """Forward function."""
        assert preds_S.shape[-2:] == preds_T.shape[-2:],'the output dim of teacher and student differ'

        if preds_S.dim () == 3:
            preds_S = preds_S.unsqueeze(1)
            preds_T = preds_T.unsqueeze(1)

        N,C,W,H = preds_S.shape

        if self.align is not None:
            preds_S = self.align(preds_S)

        softmax_pred_T = F.softmax(preds_T.view(-1,W*H)/self.tau, dim=1)
        # softmax_pred_S = F.softmax(preds_S.view(-1,W*H)/self.tau, dim=1)
        
        logsoftmax = torch.nn.LogSoftmax(dim=1)
        loss = torch.sum(- softmax_pred_T * logsoftmax(preds_S.view(-1, W * H) / self.tau)) * (self.tau ** 2)

        # loss = torch.nan_to_num(loss, posinf=0.0, neginf=0.0)
        # if self.tau < 1:
        #     loss = torch.sum( - softmax_pred_T * logsoftmax(preds_S.view(-1,W*H)/self.tau)) * (self.tau ** 2)
        # else:
        #     loss = torch.sum( - softmax_pred_T * logsoftmax(preds_S.view(-1,W*H))) * (self.tau ** 2)

        return self.loss_weight * loss / (C * N)


@DISTILL_LOSSES.register_module()
class Distill_L1Loss(nn.Module):

    """PyTorch version of `Channel-wise Distillation for Semantic Segmentation
     <https://arxiv.org/abs/2011.13256>`_.
   
    Args:
        student_channels(int): Number of channels in the student's feature map.
        teacher_channels(int): Number of channels in the teacher's feature map.
        name(str): 
        tau (float, optional): Temperature coefficient. Defaults to 1.0.
        weight (float, optional): Weight of loss.Defaults to 1.0.
        
    """
    def __init__(self,
                 name,
                 weight=1.0,
                 query=False,
                 ):
        super(Distill_L1Loss, self).__init__()
        self.loss_weight = weight
        self.query = query

    def forward(self,
                preds_S,
                preds_T,
                mask=None):
        """Forward function."""
        assert preds_S.shape[-2:] == preds_T.shape[-2:],'the output dim of teacher and student differ'
        if mask is not None:
            assert preds_S.shape[-2:] == mask.shape[-2:]
        if self.query:
            avg_factor = preds_S.shape[0]
        else:
            avg_factor = preds_S.shape[-2] * preds_S.shape[-1]
        diff = torch.abs(preds_S - preds_T)
        if mask is not None:
            diff = diff * mask
            avg_factor = mask.sum()
        avg_factor = max(avg_factor, 1.0)
        loss = torch.sum(diff)

        return self.loss_weight * loss / avg_factor

@DISTILL_LOSSES.register_module()
class Distill_L2Loss(nn.Module):

    """PyTorch version of `Channel-wise Distillation for Semantic Segmentation
     <https://arxiv.org/abs/2011.13256>`_.
   
    Args:
        student_channels(int): Number of channels in the student's feature map.
        teacher_channels(int): Number of channels in the teacher's feature map.
        name(str): 
        tau (float, optional): Temperature coefficient. Defaults to 1.0.
        weight (float, optional): Weight of loss.Defaults to 1.0.
        
    """
    def __init__(self,
                 name,
                 weight=1.0,
                 query=False,
                 ):
        super(Distill_L2Loss, self).__init__()
        self.loss_weight = weight
        self.query = query

    def forward(self,
                preds_S,
                preds_T,
                mask=None):
        """Forward function."""
        assert preds_S.shape[-2:] == preds_T.shape[-2:],'the output dim of teacher and student differ'
        if mask is not None:
            assert preds_S.shape[-2:] == mask.shape[-2:]
        if self.query:
            avg_factor = preds_S.shape[0]
        else:
            avg_factor = preds_S.shape[-2] * preds_S.shape[-1]
        diff = (preds_S - preds_T) ** 2
        if mask is not None:
            diff = diff * mask
            avg_factor = mask.sum()
        avg_factor = max(avg_factor, 1.0)
        loss = torch.sum(diff) ** 0.5

        return self.loss_weight * loss / avg_factor


@DISTILL_LOSSES.register_module()
class SpatialDistill_L2Loss(nn.Module):

    """PyTorch version of `Channel-wise Distillation for Semantic Segmentation
     <https://arxiv.org/abs/2011.13256>`_.
   
    Args:
        student_channels(int): Number of channels in the student's feature map.
        teacher_channels(int): Number of channels in the teacher's feature map.
        name(str): 
        tau (float, optional): Temperature coefficient. Defaults to 1.0.
        weight (float, optional): Weight of loss.Defaults to 1.0.
        
    """
    def __init__(self,
                 name,
                 weight=1.0,
                 tau=1.0,
                 query=False,
                 align=False,
                 ):
        super(SpatialDistill_L2Loss, self).__init__()
        self.loss_weight = weight
        self.query = query

        self.tau = tau
        self.loss_weight = weight

        if align:
            self.align = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        else:
            self.align = None
        
    def forward(self,
                preds_S,
                preds_T,
                mask=None):
        """Forward function."""
        assert preds_S.shape[-2:] == preds_T.shape[-2:],'the output dim of teacher and student differ'
        if mask is not None:
            assert preds_S.shape[-2:] == mask.shape[-2:]
        # avg_factor = preds_S.shape[-2] * preds_S.shape[-1]

        
        # self.tau = 0.25

        t_attention_mask = torch.mean(torch.abs(preds_T), [1], keepdim=True)
        size = t_attention_mask.size()
        t_attention_mask = t_attention_mask.view(preds_T.size(0), -1)
        t_attention_mask = torch.softmax(- t_attention_mask / self.tau, dim=1) * size[-1] * size[-2]
        t_attention_mask = torch.clamp(torch.nan_to_num(t_attention_mask.view(size), posinf=1.0), 0, 1)


        s_attention_mask = torch.mean(torch.abs(preds_S), [1], keepdim=True)
        size = s_attention_mask.size()
        s_attention_mask = s_attention_mask.view(preds_S.size(0), -1)
        s_attention_mask = torch.softmax(- s_attention_mask / self.tau, dim=1) * size[-1] * size[-2]
        s_attention_mask = s_attention_mask.view(size)
        s_attention_mask = torch.clamp(torch.nan_to_num(s_attention_mask.view(size), posinf=1.0), 0, 1)

        # print(s_attention_mask.min(), t_attention_mask.min(), s_attention_mask.max(), t_attention_mask.max(), size[-1] * size[-2])

        self.imshow(s_attention_mask, t_attention_mask)

        mask = s_attention_mask.detach()

        if self.align is not None:
            preds_S = self.align(preds_S.float())

        diff = (preds_S - preds_T) ** 2
        if mask is not None:
            diff = diff * mask
            avg_factor = mask.sum()
        avg_factor = max(avg_factor, 1.0)
        loss = torch.sum(diff) ** 0.5

        return self.loss_weight * loss #/ avg_factor

    def imshow(self, preds_S, preds_T):


        S = torch.clamp(preds_S,0,1).detach().cpu().numpy()
        T = torch.clamp(preds_T,0,1).detach().cpu().numpy()

        S = S[0][0]*255
        T = T[0][0]*255
        # print(S.max(), T.max(), S.shape)

        filename = np.random.randint(0,100)
        path = "./outputs/spation_attention/"
        if not os.path.exists(path):
            os.makedirs(path)
        cv2.imwrite(path+str(filename)+"_s.png", S.astype(np.uint8))
        cv2.imwrite(path+str(filename)+"_t.png", T.astype(np.uint8))

