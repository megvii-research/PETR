from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, Hook
from bisect import bisect_right


@HOOKS.register_module()
class Weighter(Hook):
    def __init__(
        self,
        steps=None,
        vals=None,
        name=None,
    ):
        self.steps = steps
        self.vals = vals
        self.name = name
        if self.name is not None:
            assert self.steps is not None
            assert self.vals is not None
            assert len(self.vals) == len(self.steps) + 1

    def before_train_iter(self, runner):
        curr_step = runner.iter
        if self.name is None:
            return
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        assert hasattr(model, self.name)
        self.steps = [s if s > 0 else runner.max_iters - s for s in self.steps]
        runner.log_buffer.output[self.name] = self.vals[
            bisect_right(self.steps, curr_step)
        ]

        setattr(model, self.name, runner.log_buffer.output[self.name])
