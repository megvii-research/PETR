# Copyright (c) Open-MMLab. All rights reserved.
import os.path as osp
import time

import torch
from torch.optim import Optimizer

import mmcv
from mmcv.runner.base_runner import BaseRunner
from mmcv.runner.checkpoint import save_checkpoint
from mmcv.runner.hooks import IterTimerHook
from mmcv.runner.utils import get_host_info
from mmcv.runner import HOOKS,Hook
from torch.nn.utils import clip_grad

@HOOKS.register_module()
class DiscriminatorOptimizerHook(Hook):

    def __init__(self, grad_clip=None):
        self.grad_clip = grad_clip

    def clip_grads(self, params):
        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return clip_grad.clip_grad_norm_(params, **self.grad_clip)

    def after_train_iter(self, runner):
        for name,optim in runner.optimizer.items():
            optim.zero_grad()
        runner.outputs['loss'].backward(retain_graph=True)
        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                        runner.outputs['num_samples'])
        for name,optim in runner.optimizer.items():
            optim.step()