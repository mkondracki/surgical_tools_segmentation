# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Code borrowed from TLC - https://www.internalfb.com/code/fbsource/fbcode/pytorch/tlc/torchtlc/loggers/tensorboard.py
import atexit
import functools
import logging
import os
import sys
import uuid
from typing import Any, Dict, Optional, Union

from hydra.utils import instantiate

from iopath.common.file_io import g_pathmgr
from numpy import ndarray
from torch import Tensor, linspace
from torch.utils.tensorboard import SummaryWriter

from training.utils.train_utils import get_machine_local_and_dist_rank, makedir

Scalar = Union[Tensor, ndarray, int, float]


def make_tensorboard_logger(log_dir: str, **writer_kwargs: Any):
    makedir(log_dir)
    summary_writer_method = SummaryWriter
    return TensorBoardLogger(
        path=log_dir, summary_writer_method=summary_writer_method, **writer_kwargs
    )
    
# def make_wandb_logger(**writer_kwargs: Any):
#     return WandbLogger(**writer_kwargs)
    


class TensorBoardWriterWrapper:
    """
    A wrapper around a SummaryWriter object.
    """

    def __init__(
        self,
        path: str,
        *args: Any,
        filename_suffix: str = None,
        summary_writer_method: Any = SummaryWriter,
        **kwargs: Any,
    ) -> None:
        """Create a new TensorBoard logger.
        On construction, the logger creates a new events file that logs
        will be written to.  If the environment variable `RANK` is defined,
        logger will only log if RANK = 0.

        NOTE: If using the logger with distributed training:
        - This logger can call collective operations
        - Logs will be written on rank 0 only
        - Logger must be constructed synchronously *after* initializing distributed process group.

        Args:
            path (str): path to write logs to
            *args, **kwargs: Extra arguments to pass to SummaryWriter
        """
        self._writer: Optional[SummaryWriter] = None
        _, self._rank = get_machine_local_and_dist_rank()
        self._path: str = path
        if self._rank == 0:
            logging.info(
                f"TensorBoard SummaryWriter instantiated. Files will be stored in: {path}"
            )
            self._writer = summary_writer_method(
                log_dir=path,
                *args,
                filename_suffix=filename_suffix or str(uuid.uuid4()),
                **kwargs,
            )
        else:
            logging.debug(
                f"Not logging meters on this host because env RANK: {self._rank} != 0"
            )
        atexit.register(self.close)

    @property
    def writer(self) -> Optional[SummaryWriter]:
        return self._writer

    @property
    def path(self) -> str:
        return self._path

    def flush(self) -> None:
        """Writes pending logs to disk."""

        if not self._writer:
            return

        self._writer.flush()

    def close(self) -> None:
        """Close writer, flushing pending logs to disk.
        Logs cannot be written after `close` is called.
        """

        if not self._writer:
            return

        self._writer.close()
        self._writer = None
        


class TensorBoardLogger(TensorBoardWriterWrapper):
    """
    A simple logger for TensorBoard.
    """

    def log_dict(self, payload: Dict[str, Scalar], step: int) -> None:
        """Add multiple scalar values to TensorBoard.

        Args:
            payload (dict): dictionary of tag name and scalar value
            step (int, Optional): step value to record
        """
        if not self._writer:
            return
        for k, v in payload.items():
            self.log(k, v, step)

    def log(self, name: str, data: Scalar, step: int) -> None:
        """Add scalar data to TensorBoard.

        Args:
            name (string): tag name used to group scalars
            data (float/int/Tensor): scalar data to log
            step (int, optional): step value to record
        """
        if not self._writer:
            return
        self._writer.add_scalar(name, data, global_step=step, new_style=True)

    def log_hparams(
        self, hparams: Dict[str, Scalar], meters: Dict[str, Scalar]
    ) -> None:
        """Add hyperparameter data to TensorBoard.

        Args:
            hparams (dict): dictionary of hyperparameter names and corresponding values
            meters (dict): dictionary of name of meter and corersponding values
        """
        if not self._writer:
            return
        self._writer.add_hparams(hparams, meters)
    
    def log_images(
        self, images: Tensor, segmentations: Tensor, gt: Tensor, step: int, max_images: int = 5
    ) -> None:
        """Log images and their segmentation outputs to TensorBoard.

        Args:
            images (Tensor): Batch of input images (shape: [B, C, H, W]).
            segmentations (Tensor): Batch of segmentation outputs (shape: [B, C, H, W]).
            step (int): Step value to record.
            max_images (int): Maximum number of images to log.
        """
        if not self._writer:
            return

        # Log a limited number of images
        assert images.dim() == 4, "Images should be a 4D tensor [B, C, H, W]"
        for i in range(images.size(0)):
            self._writer.add_image(
                f"Image_{step}", images[i].squeeze(), global_step=i
            )
        for i in range(len(segmentations)):
            masks = segmentations[i].squeeze()
            # Assign a unique value to each mask, spaced as far as possible in [0, 255]
            num_masks = masks.shape[0]
            if num_masks > 1:
                values = (linspace(0, 255, steps=num_masks + 1)[1:]).to(masks.device)
            else:
                values = Tensor([255.0], device=masks.device)
            # Combine masks into a single image with unique values
            combined_mask = (masks * values[:, None, None]).sum(dim=0, keepdim=True)
            self._writer.add_image(
                f"Segmentation_{step}", combined_mask, global_step=i
            )
        for i in range(gt.size(0)):
            gt_masks = gt[i].squeeze()
            num_gt_masks = gt_masks.shape[0]
            if num_gt_masks > 1:
                gt_values = (linspace(0, 255, steps=num_gt_masks + 1)[1:]).to(gt_masks.device)
            else:
                gt_values = Tensor([255.0], device=gt_masks.device)
            combined_gt_mask = (gt_masks * gt_values[:, None, None]).sum(dim=0, keepdim=True)
            self._writer.add_image(
            f"GT_{step}", combined_gt_mask, global_step=i
            )


class Logger:
    """
    A logger class that can interface with multiple loggers. It now supports tensorboard only for simplicity, but you can extend it with your own logger.
    """

    def __init__(self, logging_conf):
        # allow turning off TensorBoard with "should_log: false" in config
        tb_config = logging_conf.tensorboard_writer
        tb_should_log = tb_config and tb_config.pop("should_log", True)
        self.tb_logger = instantiate(tb_config) if tb_should_log else None
        
        # Wandb setup
        # wandb_config = logging_conf.wandb_writer
        # wandb_should_log = wandb_config and wandb_config.pop("should_log", True)
        # self.wandb_logger = instantiate(wandb_config) if wandb_should_log else None

    def log_dict(self, payload: Dict[str, Scalar], step: int) -> None:
        if self.tb_logger:
            self.tb_logger.log_dict(payload, step)
        # if self.wandb_logger:
        #     self.wandb_logger.log_dict(payload, step)

    def log(self, name: str, data: Scalar, step: int) -> None:
        if self.tb_logger:
            self.tb_logger.log(name, data, step)
        # if self.wandb_logger:
        #     self.wandb_logger.log(name, data, step)

    def log_hparams(
        self, hparams: Dict[str, Scalar], meters: Dict[str, Scalar]
    ) -> None:
        if self.tb_logger:
            self.tb_logger.log_hparams(hparams, meters)
        # if self.wandb_logger:
        #     self.wandb_logger.log_hparams(hparams, meters)
    



# cache the opened file object, so that different calls to `setup_logger`
# with the same file name can safely write to the same file.
@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    # we tune the buffering value so that the logs are updated
    # frequently.
    log_buffer_kb = 10 * 1024  # 10KB
    io = g_pathmgr.open(filename, mode="a", buffering=log_buffer_kb)
    atexit.register(io.close)
    return io


def setup_logging(
    name,
    output_dir=None,
    rank=0,
    log_level_primary="INFO",
    log_level_secondary="ERROR",
):
    """
    Setup various logging streams: stdout and file handlers.
    For file handlers, we only setup for the master gpu.
    """
    # get the filename if we want to log to the file as well
    log_filename = None
    if output_dir:
        makedir(output_dir)
        if rank == 0:
            log_filename = f"{output_dir}/log.txt"

    logger = logging.getLogger(name)
    logger.setLevel(log_level_primary)

    # create formatter
    FORMAT = "%(levelname)s %(asctime)s %(filename)s:%(lineno)4d: %(message)s"
    formatter = logging.Formatter(FORMAT)

    # Cleanup any existing handlers
    for h in logger.handlers:
        logger.removeHandler(h)
    logger.root.handlers = []

    # setup the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if rank == 0:
        console_handler.setLevel(log_level_primary)
    else:
        console_handler.setLevel(log_level_secondary)

    # we log to file as well if user wants
    if log_filename and rank == 0:
        file_handler = logging.StreamHandler(_cached_log_stream(log_filename))
        file_handler.setLevel(log_level_primary)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logging.root = logger
    


def shutdown_logging():
    """
    After training is done, we ensure to shut down all the logger streams.
    """
    logging.info("Shutting down loggers...")
    handlers = logging.root.handlers
    for handler in handlers:
        handler.close()
        
