import warnings
from collections import Counter, Mapping, Sequence
from numbers import Number
from typing import Dict, List

import numpy as np
import torch
from mmdet.core.mask.structures import BitmapMasks
from torch.nn import functional as F

_step_counter = Counter()


def list_concat(data_list: List[list]):
    if isinstance(data_list[0], torch.Tensor):
        return torch.cat(data_list)
    else:
        endpoint = [d for d in data_list[0]]

        for i in range(1, len(data_list)):
            endpoint.extend(data_list[i])
        return endpoint


def sequence_concat(a, b):
    if isinstance(a, Sequence) and isinstance(b, Sequence):
        return a + b
    else:
        return None


def dict_concat(dicts: List[Dict[str, list]]):
    return {k: list_concat([d[k] for d in dicts]) for k in dicts[0].keys()}


def dict_fuse(obj_list, reference_obj):
    if isinstance(reference_obj, torch.Tensor):
        return torch.stack(obj_list)
    return obj_list


def dict_select(dict1: Dict[str, list], key: str, value: str):
    flag = [v == value for v in dict1[key]]
    return {
        k: dict_fuse([vv for vv, ff in zip(v, flag) if ff], v) for k, v in dict1.items()
    }


def dict_split(dict1, key):
    group_names = list(set(dict1[key]))
    dict_groups = {k: dict_select(dict1, key, k) for k in group_names}

    return dict_groups


def dict_sum(a, b):
    if isinstance(a, dict):
        assert isinstance(b, dict)
        return {k: dict_sum(v, b[k]) for k, v in a.items()}
    elif isinstance(a, list):
        assert len(a) == len(b)
        return [dict_sum(aa, bb) for aa, bb in zip(a, b)]
    else:
        return a + b


def zero_like(tensor_pack, prefix=""):
    if isinstance(tensor_pack, Sequence):
        return [zero_like(t) for t in tensor_pack]
    elif isinstance(tensor_pack, Mapping):
        return {prefix + k: zero_like(v) for k, v in tensor_pack.items()}
    elif isinstance(tensor_pack, torch.Tensor):
        return tensor_pack.new_zeros(tensor_pack.shape)
    elif isinstance(tensor_pack, np.ndarray):
        return np.zeros_like(tensor_pack)
    else:
        warnings.warn("Unexpected data type {}".format(type(tensor_pack)))
        return 0


def pad_stack(tensors, shape, pad_value=255):
    tensors = torch.stack(
        [
            F.pad(
                tensor,
                pad=[0, shape[1] - tensor.shape[1], 0, shape[0] - tensor.shape[0]],
                value=pad_value,
            )
            for tensor in tensors
        ]
    )
    return tensors


def result2bbox(result):
    num_class = len(result)

    bbox = np.concatenate(result)
    if bbox.shape[0] == 0:
        label = np.zeros(0, dtype=np.uint8)
    else:
        label = np.concatenate(
            [[i] * len(result[i]) for i in range(num_class) if len(result[i]) > 0]
        ).reshape((-1,))
    return bbox, label


def result2mask(result):
    num_class = len(result)
    mask = [np.stack(result[i]) for i in range(num_class) if len(result[i]) > 0]
    if len(mask) > 0:
        mask = np.concatenate(mask)
    else:
        mask = np.zeros((0, 1, 1))
    return BitmapMasks(mask, mask.shape[1], mask.shape[2]), None


def sequence_mul(obj, multiplier):
    if isinstance(obj, Sequence):
        return [o * multiplier for o in obj]
    else:
        return obj * multiplier


def is_match(word, word_list):
    for keyword in word_list:
        if keyword in word:
            return True
    return False


def weighted_loss(loss: dict, weight, ignore_keys=[], warmup=0):
    _step_counter["weight"] += 1
    lambda_weight = (
        lambda x: x * (_step_counter["weight"] - 1) / warmup
        if _step_counter["weight"] <= warmup
        else x
    )
    if isinstance(weight, Mapping):
        for k, v in weight.items():
            for name, loss_item in loss.items():
                if (k in name) and ("loss" in name):
                    loss[name] = sequence_mul(loss[name], lambda_weight(v))
    elif isinstance(weight, Number):
        for name, loss_item in loss.items():
            if "loss" in name:
                if not is_match(name, ignore_keys):
                    loss[name] = sequence_mul(loss[name], lambda_weight(weight))
                else:
                    loss[name] = sequence_mul(loss[name], 0.0)
    else:
        raise NotImplementedError()
    return loss
