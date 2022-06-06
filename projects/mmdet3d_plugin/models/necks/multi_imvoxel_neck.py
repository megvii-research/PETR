# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.cnn import ConvModule
from torch import nn

from mmdet.models import NECKS


@NECKS.register_module()
class Multi_ImVoxelNeck(nn.Module):
    """Neck for ImVoxelNet outdoor scenario.

    Args:
        in_channels (int): Input channels of multi-scale feature map.
        out_channels (int): Output channels of multi-scale feature map.
    """

    def __init__(self, in_channels, out_channels, stride=[2, 2, 2]):
        super().__init__()
        self.stage0 = nn.Sequential(
            ResModule(in_channels),
            ConvModule(
                in_channels=in_channels,
                out_channels=in_channels * 2,
                kernel_size=3,
                stride=stride[0],
                padding=1,
                conv_cfg=dict(type='Conv3d'),
                norm_cfg=dict(type='BN3d'),
                act_cfg=dict(type='ReLU', inplace=True)))
        self.stage1 = nn.Sequential(
            ResModule(in_channels * 2),
            ConvModule(
                in_channels=in_channels * 2,
                out_channels=in_channels * 4,
                kernel_size=3,
                stride=stride[1],
                padding=1,
                conv_cfg=dict(type='Conv3d'),
                norm_cfg=dict(type='BN3d'),
                act_cfg=dict(type='ReLU', inplace=True)))
        self.stage2 = nn.Sequential(
            ResModule(in_channels * 4),
            ConvModule(
                in_channels=in_channels * 4,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride[2],
                padding=1,
                conv_cfg=dict(type='Conv3d'),
                norm_cfg=dict(type='BN3d'),
                act_cfg=dict(type='ReLU', inplace=True)))

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): of shape (N, C_in, N_x, N_y, N_z).

        Returns:
            list[torch.Tensor]: of shape (N, C_out, N_y, N_x).
        """
        out = []
        for i in range(3):
            stage_layer = getattr(self, f'stage{i}')
            x = stage_layer(x)
            
            # s = x.mean(-1).transpose(-1, -2)
            # print(s.size())
            out.append(x.mean(-1).transpose(-1, -2))

        # Anchor3DHead axis order is (y, x).
        return out

    def init_weights(self):
        """Initialize weights of neck."""
        pass


class ResModule(nn.Module):
    """3d residual block for ImVoxelNeck.

    Args:
        n_channels (int): Input channels of a feature map.
    """

    def __init__(self, n_channels):
        super().__init__()
        self.conv0 = ConvModule(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=3,
            padding=1,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='BN3d'),
            act_cfg=dict(type='ReLU', inplace=True))
        self.conv1 = ConvModule(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=3,
            padding=1,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='BN3d'),
            act_cfg=None)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): of shape (N, C, N_x, N_y, N_z).

        Returns:
            torch.Tensor: 5d feature map.
        """
        identity = x
        x = self.conv0(x)
        x = self.conv1(x)
        x = identity + x
        x = self.activation(x)
        return x
