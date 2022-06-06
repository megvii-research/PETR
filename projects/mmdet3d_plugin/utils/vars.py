import re
from typing import Union

pattern = re.compile("\$\{[a-zA-Z\d_.]*\}")


def get_value(cfg: dict, chained_key: str):
    keys = chained_key.split(".")
    if len(keys) == 1:
        return cfg[keys[0]]
    else:
        return get_value(cfg[keys[0]], ".".join(keys[1:]))


def resolve(cfg: Union[dict, list], base=None):
    if base is None:
        base = cfg
    if isinstance(cfg, dict):
        return {k: resolve(v, base) for k, v in cfg.items()}
    elif isinstance(cfg, list):
        return [resolve(v, base) for v in cfg]
    elif isinstance(cfg, tuple):
        return tuple([resolve(v, base) for v in cfg])
    elif isinstance(cfg, str):
        # process
        var_names = pattern.findall(cfg)
        if len(var_names) == 1 and len(cfg) == len(var_names[0]):
            return get_value(base, var_names[0][2:-1])
        else:
            vars = [get_value(base, name[2:-1]) for name in var_names]
            for name, var in zip(var_names, vars):
                cfg = cfg.replace(name, str(var))
            return cfg
    else:
        return cfg
