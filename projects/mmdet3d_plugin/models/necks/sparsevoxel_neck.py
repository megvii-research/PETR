# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner import auto_fp16
from torch import nn as nn
from mmdet3d.ops import SparseBasicBlock, make_sparse_convmodule
from mmdet3d.ops import spconv as spconv
from mmcv.cnn import ConvModule
from torch import nn

from mmdet.models import NECKS

class ResModule(spconv.SparseModule):
    """3d residual block for ImVoxelNeck.

    Args:
        n_channels (int): Input channels of a feature map.
    """

    def __init__(self, 
                 n_channels, 
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 block_name = "res",
                ):
        super().__init__()
        self.conv0 = make_sparse_convmodule(
                n_channels,
                n_channels,
                3,
                norm_cfg=norm_cfg,
                padding=1,
                indice_key=block_name+'_subm1',
                conv_type='SubMConv3d')
        self.conv1 = make_sparse_convmodule(
                n_channels,
                n_channels,
                3,
                norm_cfg=norm_cfg,
                padding=1,
                indice_key=block_name+'_subm1',
                conv_type='SubMConv3d')
        self.activation = nn.ReLU(inplace=True)

    # @auto_fp16(apply_to=('x', ))
    def forward(self, x):
        """Forward function.
        Args:
            x (torch.Tensor): of shape (N, C, N_z, N_y, N_x).

        Returns:
            torch.Tensor: 5d feature map.
        """
        identity = x

        x = self.conv0(x)
        x = self.conv1(x)

        features = identity.features + x.features
        features = self.activation(features)
        indices = x.indices
        spatial_shape = x.spatial_shape
        batch_size = x.batch_size
        x = spconv.SparseConvTensor(features, indices, spatial_shape, batch_size)

        return x

@NECKS.register_module()
class SparseImVoxelNeck(nn.Module):
    r"""Sparse encoder for SECOND and Part-A2.

    Args:
        in_channels (int): The number of input channels.
        sparse_shape (list[int]): The sparse shape of input tensor.
        order (list[str]): Order of conv module. Defaults to ('conv',
            'norm', 'act').
        norm_cfg (dict): Config of normalization layer. Defaults to
            dict(type='BN1d', eps=1e-3, momentum=0.01).
        base_channels (int): Out channels for conv_input layer.
            Defaults to 16.
        output_channels (int): Out channels for conv_out layer.
            Defaults to 128.
        encoder_channels (tuple[tuple[int]]):
            Convolutional channels of each encode block.
        encoder_paddings (tuple[tuple[int]]): Paddings of each encode block.
            Defaults to ((16, ), (32, 32, 32), (64, 64, 64), (64, 64, 64)).
        block_type (str): Type of the block to use. Defaults to 'conv_module'.
    """

    def __init__(self,
                 in_channels,
                 sparse_shape=None,
                 order=('conv', 'norm', 'act'),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 base_channels=64,
                 output_channels=128,
                 encoder_channels=((64, 64), (64, 64), (64, 64), (64, 64)),
                 encoder_paddings=((1, 1), (1, 1), (1, 1), (1, 1)),
                 block_type='conv_module'):
        super().__init__()
        assert block_type in ['conv_module', 'basicblock']
        self.sparse_shape = sparse_shape
        self.in_channels = in_channels
        self.order = order
        self.base_channels = base_channels
        self.output_channels = output_channels
        self.encoder_channels = encoder_channels
        self.encoder_paddings = encoder_paddings
        self.stage_num = len(self.encoder_channels)
        self.fp16_enabled = False
        # Spconv init all weight on its own

        assert isinstance(order, tuple) and len(order) == 3
        assert set(order) == {'conv', 'norm', 'act'}

        if self.order[0] != 'conv':  # pre activate
            self.conv_input = make_sparse_convmodule(
                in_channels,
                self.base_channels,
                3,
                norm_cfg=norm_cfg,
                padding=1,
                indice_key='subm1',
                conv_type='SubMConv3d',
                order=('conv', ))
        else:  # post activate
            self.conv_input = make_sparse_convmodule(
                in_channels,
                self.base_channels,
                3,
                norm_cfg=norm_cfg,
                padding=1,
                indice_key='subm1',
                conv_type='SubMConv3d')

        self.model = spconv.SparseSequential(
            ResModule(self.base_channels, block_name = "res0",),
            make_sparse_convmodule(
                self.base_channels,
                self.base_channels,
                kernel_size=3,
                stride=(2, 1, 1),
                norm_cfg=norm_cfg,
                padding=1,
                indice_key='spconv_down0',
                conv_type='SparseConv3d'),
            ResModule(self.base_channels, block_name = "res1",),
            make_sparse_convmodule(
                self.base_channels,
                self.base_channels,
                kernel_size=3,
                stride=(2, 1, 1),
                norm_cfg=norm_cfg,
                padding=1,
                indice_key='spconv_down1',
                conv_type='SparseConv3d'),
            ResModule(self.base_channels, block_name = "res2",),
            make_sparse_convmodule(
                self.base_channels,
                self.base_channels,
                kernel_size=3,
                stride=(2, 1, 1),
                norm_cfg=norm_cfg,
                padding=1,
                indice_key='spconv_down2',
                conv_type='SparseConv3d'),
            ResModule(self.base_channels, block_name = "res3",),
            make_sparse_convmodule(
                self.base_channels,
                output_channels,
                kernel_size=3,
                stride=(2, 1, 1),
                norm_cfg=norm_cfg,
                padding=(0, 1, 1),
                indice_key='spconv_down3',
                conv_type='SparseConv3d'),
            )

        # self.conv_out = make_sparse_convmodule(
        #     in_channels,
        #     in_channels,
        #     kernel_size=(3, 1, 1),
        #     stride=(2, 1, 1),
        #     norm_cfg=norm_cfg,
        #     padding=0,
        #     indice_key='spconv_down2',
        #     conv_type='SparseConv3d')

    # @auto_fp16(apply_to=('voxel_features', ))
    def forward(self, voxel_features, coors, sparse_shape, batch_size):
        """Forward of SparseEncoder.

        Args:
            voxel_features (torch.float32): Voxel features in shape (N, C).
            coors (torch.int32): Coordinates in shape (N, 4), \
                the columns in the order of (batch_idx, z_idx, y_idx, x_idx).
            batch_size (int): Batch size.

        Returns:
            dict: Backbone features.
        """
        coors = coors.int()
        input_sp_tensor = spconv.SparseConvTensor(voxel_features, coors,
                                                  sparse_shape,
                                                  batch_size)
        input_sp_tensor =  self.conv_input(input_sp_tensor)
        out = self.model(input_sp_tensor)
        # print(out.features.size())
        # out = self.conv_out(encode_features[-1])
        spatial_features = out.dense()

        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)

        return spatial_features

