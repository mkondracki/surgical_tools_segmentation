import logging
import os

import torch
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

import sam2

def build_dataset(
    config_file,
    **kwargs,
):
    # Read config and init model
    cfg = compose(config_name=config_file)
    OmegaConf.resolve(cfg)
    dataset = instantiate(cfg.data.val)
    transform = instantiate(cfg.vos.val_transforms)
    return dataset, cfg.dataset


def build_transform(
    config_file,
    **kwargs,
):
    # Read config and init model
    cfg = compose(config_name=config_file)
    OmegaConf.resolve(cfg)
    transform = instantiate(cfg.vos.val_transforms)
    return transform, cfg.scratch.resolution