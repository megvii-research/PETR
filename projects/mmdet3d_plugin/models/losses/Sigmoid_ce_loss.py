# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

import math
import warnings

import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.core import bbox_overlaps
from mmdet.models.builder import LOSSES
from mmdet.models.losses.utils import weighted_loss
import numpy as np



@LOSSES.register_module()
class Sigmoid_ce_loss(nn.Module):

    def __init__(self,  loss_weight=1.0):
        super(Sigmoid_ce_loss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self,inputs,
        targets,
        ):
        """Forward function to calculate accuracy.

        Args:
            pred (torch.Tensor): Prediction of models.
            target (torch.Tensor): Target for each prediction.

        Returns:
            tuple[float]: The accuracies under different topk criterions.
        """
        # inputs=inputs.sigmoid()
        pos_weight = (targets == 0).float().sum(dim=1) / (targets == 1).float().sum(dim=1).clamp(min=1.0)
        pos_weight=pos_weight.unsqueeze(1)
        weight_loss=targets*pos_weight+(1-targets)
        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="mean",weight=weight_loss)
        return self.loss_weight*loss