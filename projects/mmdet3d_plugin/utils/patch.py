import glob
import os
import os.path as osp
import shutil
import types

from mmcv.runner import BaseRunner, EpochBasedRunner, IterBasedRunner
from mmcv.utils import Config

from .signature import parse_method_info
from .vars import resolve


def find_latest_checkpoint(path, ext="pth"):
    if not osp.exists(path):
        return None
    if osp.exists(osp.join(path, f"latest.{ext}")):
        return osp.join(path, f"latest.{ext}")

    checkpoints = glob.glob(osp.join(path, f"*.{ext}"))
    if len(checkpoints) == 0:
        return None
    latest = -1
    latest_path = None
    for checkpoint in checkpoints:
        count = int(osp.basename(checkpoint).split("_")[-1].split(".")[0])
        if count > latest:
            latest = count
            latest_path = checkpoint
    return latest_path


def patch_checkpoint(runner: BaseRunner):
    # patch save_checkpoint
    old_save_checkpoint = runner.save_checkpoint
    params = parse_method_info(old_save_checkpoint)
    default_tmpl = params["filename_tmpl"].default

    def save_checkpoint(self, out_dir, **kwargs):
        create_symlink = kwargs.get("create_symlink", True)
        filename_tmpl = kwargs.get("filename_tmpl", default_tmpl)
        # create_symlink
        kwargs.update(create_symlink=False)
        old_save_checkpoint(out_dir, **kwargs)
        if create_symlink:
            dst_file = osp.join(out_dir, "latest.pth")
            if isinstance(self, EpochBasedRunner):
                filename = filename_tmpl.format(self.epoch + 1)
            elif isinstance(self, IterBasedRunner):
                filename = filename_tmpl.format(self.iter + 1)
            else:
                raise NotImplementedError()
            filepath = osp.join(out_dir, filename)
            shutil.copy(filepath, dst_file)

    runner.save_checkpoint = types.MethodType(save_checkpoint, runner)
    return runner


def patch_runner(runner):
    runner = patch_checkpoint(runner)
    return runner


def setup_env(cfg):
    os.environ["WORK_DIR"] = cfg.work_dir


def patch_config(cfg):

    cfg_dict = super(Config, cfg).__getattribute__("_cfg_dict").to_dict()
    cfg_dict["cfg_name"] = osp.splitext(osp.basename(cfg.filename))[0]
    cfg_dict = resolve(cfg_dict)
    cfg = Config(cfg_dict, filename=cfg.filename)
    # wrap for semi
    if cfg.get("semi_wrapper", None) is not None:
        cfg.model = cfg.semi_wrapper
        cfg.pop("semi_wrapper")
    # enable environment variables
    setup_env(cfg)
    return cfg
