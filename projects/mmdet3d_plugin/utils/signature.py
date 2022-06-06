import inspect


def parse_method_info(method):
    sig = inspect.signature(method)
    params = sig.parameters
    return params
