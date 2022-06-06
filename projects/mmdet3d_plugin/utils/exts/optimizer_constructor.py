import warnings

import torch
from torch.nn import GroupNorm, LayerNorm

from mmcv.utils import _BatchNorm, _InstanceNorm, build_from_cfg
from mmcv.utils.ext_loader import check_ops_exist
from mmcv.runner.optimizer.builder import OPTIMIZER_BUILDERS, OPTIMIZERS
from mmcv.runner.optimizer import DefaultOptimizerConstructor


@OPTIMIZER_BUILDERS.register_module()
class NamedOptimizerConstructor(DefaultOptimizerConstructor):
    """Main difference to default constructor:

    1) Add name to parame groups
    """

    def add_params(self, params, module, prefix="", is_dcn_module=None):
        """Add all parameters of module to the params list.

        The parameters of the given module will be added to the list of param
        groups, with specific rules defined by paramwise_cfg.

        Args:
            params (list[dict]): A list of param groups, it will be modified
                in place.
            module (nn.Module): The module to be added.
            prefix (str): The prefix of the module
            is_dcn_module (int|float|None): If the current module is a
                submodule of DCN, `is_dcn_module` will be passed to
                control conv_offset layer's learning rate. Defaults to None.
        """
        # get param-wise options
        custom_keys = self.paramwise_cfg.get("custom_keys", {})
        # first sort with alphabet order and then sort with reversed len of str
        sorted_keys = sorted(sorted(custom_keys.keys()), key=len, reverse=True)

        bias_lr_mult = self.paramwise_cfg.get("bias_lr_mult", 1.0)
        bias_decay_mult = self.paramwise_cfg.get("bias_decay_mult", 1.0)
        norm_decay_mult = self.paramwise_cfg.get("norm_decay_mult", 1.0)
        dwconv_decay_mult = self.paramwise_cfg.get("dwconv_decay_mult", 1.0)
        bypass_duplicate = self.paramwise_cfg.get("bypass_duplicate", False)
        dcn_offset_lr_mult = self.paramwise_cfg.get("dcn_offset_lr_mult", 1.0)

        # special rules for norm layers and depth-wise conv layers
        is_norm = isinstance(module, (_BatchNorm, _InstanceNorm, GroupNorm, LayerNorm))
        is_dwconv = (
            isinstance(module, torch.nn.Conv2d) and module.in_channels == module.groups
        )

        for name, param in module.named_parameters(recurse=False):
            param_group = {"params": [param], "name": f"{prefix}.{name}"}
            if not param.requires_grad:
                params.append(param_group)
                continue
            if bypass_duplicate and self._is_in(param_group, params):
                warnings.warn(
                    f"{prefix} is duplicate. It is skipped since "
                    f"bypass_duplicate={bypass_duplicate}"
                )
                continue
            # if the parameter match one of the custom keys, ignore other rules
            is_custom = False
            for key in sorted_keys:
                if key in f"{prefix}.{name}":
                    is_custom = True
                    lr_mult = custom_keys[key].get("lr_mult", 1.0)
                    param_group["lr"] = self.base_lr * lr_mult
                    if self.base_wd is not None:
                        decay_mult = custom_keys[key].get("decay_mult", 1.0)
                        param_group["weight_decay"] = self.base_wd * decay_mult
                    break

            if not is_custom:
                # bias_lr_mult affects all bias parameters
                # except for norm.bias dcn.conv_offset.bias
                if name == "bias" and not (is_norm or is_dcn_module):
                    param_group["lr"] = self.base_lr * bias_lr_mult

                if (
                    prefix.find("conv_offset") != -1
                    and is_dcn_module
                    and isinstance(module, torch.nn.Conv2d)
                ):
                    # deal with both dcn_offset's bias & weight
                    param_group["lr"] = self.base_lr * dcn_offset_lr_mult

                # apply weight decay policies
                if self.base_wd is not None:
                    # norm decay
                    if is_norm:
                        param_group["weight_decay"] = self.base_wd * norm_decay_mult
                    # depth-wise conv
                    elif is_dwconv:
                        param_group["weight_decay"] = self.base_wd * dwconv_decay_mult
                    # bias lr and decay
                    elif name == "bias" and not is_dcn_module:
                        # TODO: current bias_decay_mult will have affect on DCN
                        param_group["weight_decay"] = self.base_wd * bias_decay_mult
            params.append(param_group)

        if check_ops_exist():
            from mmcv.ops import DeformConv2d, ModulatedDeformConv2d

            is_dcn_module = isinstance(module, (DeformConv2d, ModulatedDeformConv2d))
        else:
            is_dcn_module = False
        for child_name, child_mod in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            self.add_params(
                params, child_mod, prefix=child_prefix, is_dcn_module=is_dcn_module
            )
