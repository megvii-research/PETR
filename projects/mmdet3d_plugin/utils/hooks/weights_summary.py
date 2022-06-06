import os.path as osp

import torch.distributed as dist
from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, Hook
from ..logger import get_root_logger
from prettytable import PrettyTable


def bool2str(input):
    if input:
        return "Y"
    else:
        return "N"


def unknown():
    return "-"


def shape_str(size):
    size = [str(s) for s in size]
    return "X".join(size)


def min_max_str(input):
    return "Min:{:.3f} Max:{:.3f}".format(input.min(), input.max())


def construct_params_dict(input):
    assert isinstance(input, list)
    param_dict = {}
    for group in input:
        if "name" in group:
            param_dict[group["name"]] = group
    return param_dict


def max_match_sub_str(strs, sub_str):
    # find most related str for sub_str
    matched = None
    for child in strs:
        if len(child) <= len(sub_str):
            if child == sub_str:
                return child
            elif sub_str[: len(child)] == child:
                if matched is None or len(matched) < len(child):
                    matched = child
    return matched


def get_optim(optimizer, params_dict, name, key):
    rel_name = max_match_sub_str(list(params_dict.keys()), name)
    if rel_name is not None:
        return params_dict[rel_name][key]
    else:
        if key in optimizer.defaults:
            return optimizer.defaults[key]


@HOOKS.register_module()
class WeightSummary(Hook):
    def before_run(self, runner):
        if runner.rank != 0:
            return
        if is_module_wrapper(runner.model):
            model = runner.model.module
        else:
            model = runner.model
        weight_summaries = self.collect_model_info(model, optimizer=runner.optimizer)
        logger = get_root_logger()
        logger.info(weight_summaries)

    @staticmethod
    def collect_model_info(model, optimizer=None, rich_text=False):
        param_groups = None
        if optimizer is not None:
            param_groups = construct_params_dict(optimizer.param_groups)

        if not rich_text:
            table = PrettyTable(
                ["Name", "Optimized", "Shape", "Value Scale [Min,Max]", "Lr", "Wd"]
            )
            for name, param in model.named_parameters():
                table.add_row(
                    [
                        name,
                        bool2str(param.requires_grad),
                        shape_str(param.size()),
                        min_max_str(param),
                        unknown()
                        if param_groups is None
                        else get_optim(optimizer, param_groups, name, "lr"),
                        unknown()
                        if param_groups is None
                        else get_optim(optimizer, param_groups, name, "weight_decay"),
                    ]
                )
            return "\n" + table.get_string(title="Model Information")
        else:
            pass
