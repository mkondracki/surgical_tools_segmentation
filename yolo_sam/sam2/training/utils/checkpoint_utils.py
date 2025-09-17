# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import fnmatch
import logging
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
import torch
import torch.nn as nn
from iopath.common.file_io import g_pathmgr
from torch.jit._script import RecursiveScriptModule


def unix_pattern_to_parameter_names(
    constraints: List[str], all_parameter_names: Sequence[str]
) -> Union[None, Set[str]]:
    """
    Go through the list of parameter names and select those that match
    any of the provided constraints
    """
    parameter_names = []
    for param_name in constraints:
        matching_parameters = set(fnmatch.filter(all_parameter_names, param_name))
        assert (
            len(matching_parameters) > 0
        ), f"param_names {param_name} don't match any param in the given names."
        parameter_names.append(matching_parameters)
    return set.union(*parameter_names)


def filter_params_matching_unix_pattern(
    patterns: List[str], state_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Remove from the state dictionary the parameters matching the provided unix patterns

    Args:
        patterns: the list of unix patterns to exclude
        state_dict: the dictionary to filter

    Returns:
        A new state dictionary
    """
    if len(patterns) == 0:
        return {}

    all_keys = list(state_dict.keys())
    included_keys = unix_pattern_to_parameter_names(patterns, all_keys)
    return {k: state_dict[k] for k in included_keys}


def exclude_params_matching_unix_pattern(
    patterns: List[str], state_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Remove from the state dictionary the parameters matching the provided unix patterns

    Args:
        patterns: the list of unix patterns to exclude
        state_dict: the dictionary to filter

    Returns:
        A new state dictionary
    """
    if len(patterns) == 0:
        return state_dict

    all_keys = list(state_dict.keys())
    excluded_keys = unix_pattern_to_parameter_names(patterns, all_keys)
    return {k: v for k, v in state_dict.items() if k not in excluded_keys}


def _get_state_dict_summary(state_dict: Dict[str, torch.Tensor]):
    keys = []
    trace = []
    for k, v in state_dict.items():
        keys.append(k)
        trace.append(v.sum().item())
    trace = np.array(trace)[np.argsort(keys)]
    return trace


def assert_skipped_parameters_are_frozen(model: nn.Module, patterns: List[str]):
    """
    Verifies that all the parameters matching the provided patterns
    are frozen - this acts as a safeguard when ignoring parameter
    when saving checkpoints - if the parameters are in fact trainable
    """
    if not patterns:
        return

    frozen_state_dict = filter_params_matching_unix_pattern(
        patterns=patterns, state_dict=model.state_dict()
    )
    non_frozen_keys = {
        n
        for n, p in model.named_parameters()
        if n in frozen_state_dict and p.requires_grad
    }
    if non_frozen_keys:
        raise ValueError(
            f"Parameters excluded with `skip_saving_parameters` should be frozen: {non_frozen_keys}"
        )


@contextlib.contextmanager
def with_check_parameter_frozen(
    model: nn.Module, patterns: List[str], disabled: bool = True
):
    """
    Context manager that inspects a model surrounding a piece of code
    and verifies if the model has been updated by this piece of code

    The function will raise an exception if the model has been updated
    on at least one of the parameter that matches one of the pattern

    Args:
        model: the model that might have been updated
        patterns: for the parameters we want to observe
        allowed:
    """
    if not patterns or disabled:
        yield
        return

    frozen_state_dict = filter_params_matching_unix_pattern(
        patterns=patterns, state_dict=model.state_dict()
    )
    summary_before = _get_state_dict_summary(frozen_state_dict)

    yield

    frozen_state_dict = filter_params_matching_unix_pattern(
        patterns=patterns, state_dict=model.state_dict()
    )
    summary_after = _get_state_dict_summary(frozen_state_dict)

    if not np.allclose(summary_before, summary_after, atol=1e-6):
        raise ValueError(
            f"""
            The `model_weight_initializer` has initialized parameters frozen with `skip_saving_parameters`.
            You can resolve this error by either initializing those parameters from within the model definition
            or using the flag `trainer.checkpoint.initialize_after_preemption` to True.
        """
        )


class CkptExcludeKernel:
    """
    Removes the keys from the given model state_dict that match the key_pattern.

    Args:
        key_pattern: Patterns used to select the keys in the state_dict
            that are eligible for this kernel.
    """

    def __init__(self, key_pattern: List[str]):
        self.key_pattern = key_pattern

    def __call__(self, state_dict: Dict):
        """
        Args:
            state_dict: A dictionary representing the given checkpoint's state dict.
        """
        if len(self.key_pattern) == 0:
            return state_dict
        exclude_keys = unix_pattern_to_parameter_names(
            self.key_pattern, state_dict.keys()
        )
        return {k: v for k, v in state_dict.items() if k not in exclude_keys}


def load_checkpoint(
    path_list: List[str],
    pick_recursive_keys: Optional[List[str]] = None,
    map_location: str = "cpu",
) -> Any:
    """
    Loads a checkpoint from the specified path.

    Args:
        path_list: A list of paths which contain the checkpoint. Each element
            is tried (in order) until a file that exists is found. That file is then
            used to read the checkpoint.
        pick_recursive_keys: Picks sub dicts from the loaded checkpoint if not None.
            For pick_recursive_keys = ["a", "b"], will return checkpoint_dict["a"]["b"]
        map_location (str): a function, torch.device, string or a dict specifying how to
            remap storage locations

    Returns: Model with the matchin pre-trained weights loaded.
    """
    path_exists = False
    for path in path_list:
        if g_pathmgr.exists(path):
            path_exists = True
            break

    if not path_exists:
        raise ValueError(f"No path exists in {path_list}")

    with g_pathmgr.open(path, "rb") as f:
        checkpoint = torch.load(f, map_location=map_location, weights_only=False)

    logging.info(f"Loaded checkpoint from {path}")
    if pick_recursive_keys is not None:
        for key in pick_recursive_keys:
            checkpoint = checkpoint[key]
    return checkpoint


def get_state_dict(checkpoint, ckpt_state_dict_keys):
    if isinstance(checkpoint, RecursiveScriptModule):
        # This is a torchscript JIT model
        return checkpoint.state_dict()
    if not ckpt_state_dict_keys:
        # No state dict keys provided, return the checkpoint as is
        return checkpoint
    pre_train_dict = checkpoint
    for i, key in enumerate(ckpt_state_dict_keys):
        if (isinstance(pre_train_dict, Mapping) and key not in pre_train_dict) or (
            isinstance(pre_train_dict, Sequence) and key >= len(pre_train_dict)
        ):
            key_str = (
                '["' + '"]["'.join(list(map(ckpt_state_dict_keys[:i], str))) + '"]'
            )
            raise KeyError(
                f"'{key}' not found in checkpoint{key_str} "
                f"with keys: {pre_train_dict.keys()}"
            )
        pre_train_dict = pre_train_dict[key]
    return pre_train_dict


def load_checkpoint_and_apply_kernels(
    checkpoint_path: str,
    checkpoint_kernels: List[Callable] = None,
    ckpt_state_dict_keys: Tuple[str] = ("state_dict",),
    map_location: str = "cpu",
) -> nn.Module:
    """
    Performs checkpoint loading with a variety of pre-processing kernel applied in
    sequence.

    Args:
        checkpoint_path (str): Path to the checkpoint.
        checkpoint_kernels List(Callable): A list of checkpoint processing kernels
            to apply in the specified order. Supported kernels include `CkptIncludeKernel`,
            `CkptExcludeKernel`, etc. These kernels are applied in the
            given order.
        ckpt_state_dict_keys (str): Keys containing the model state dict.
        map_location (str): a function, torch.device, string or a dict specifying how to
            remap storage locations

    Returns: Model with the matchin pre-trained weights loaded.
    """
    assert g_pathmgr.exists(checkpoint_path), "Checkpoint '{}' not found".format(
        checkpoint_path
    )

    # Load the checkpoint on CPU to avoid GPU mem spike.
    with g_pathmgr.open(checkpoint_path, "rb") as f:
        checkpoint = torch.load(f, map_location=map_location, weights_only=True)

    pre_train_dict = get_state_dict(checkpoint, ckpt_state_dict_keys)

    # Not logging into info etc since it's a huge log
    logging.debug(
        "Loaded Checkpoint State Dict pre-kernel application: %s"
        % str(", ".join(list(pre_train_dict.keys())))
    )
    # Apply kernels
    if checkpoint_kernels is not None:
        for f in checkpoint_kernels:
            pre_train_dict = f(state_dict=pre_train_dict)

    logging.debug(
        "Loaded Checkpoint State Dict Post-kernel application %s"
        % str(", ".join(list(pre_train_dict.keys())))
    )

    return pre_train_dict


def load_img_enc_checkpoint_and_apply_kernels(
    checkpoint_path: str,
    img_enc_key: str = "image_encoder",
    checkpoint_kernels: List[Callable] = None,
    ckpt_state_dict_keys: Tuple[str] = ("state_dict",),
    map_location: str = "cpu",
    strict: bool = True,
    ignore_missing_keys: List[str] = None,
    ignore_unexpected_keys: List[str] = None,
    model: nn.Module = None,
    weights_only: bool=True
) -> nn.Module:
    """
    Loads a checkpoint and applies kernels to extract and load only the image encoder's state dict.

    Args:
        checkpoint_path (str): Path to the checkpoint.
        img_enc_key (str): The key in the checkpoint corresponding to the image encoder.
        checkpoint_kernels (List[Callable]): A list of checkpoint processing kernels to apply.
        ckpt_state_dict_keys (Tuple[str]): Keys containing the model state dict.
        map_location (str): Specifies how to remap storage locations.
        strict (bool): Raise if the state_dict has missing state keys.
        ignore_missing_keys (List[str]): Unix pattern of keys to ignore for missing keys.
        ignore_unexpected_keys (List[str]): Unix pattern of keys to ignore for unexpected keys.
        model (nn.Module): The model to load the image encoder weights into.

    Returns:
        nn.Module: Model with the image encoder's pre-trained weights loaded.
    """
    # Load the checkpoint
    assert g_pathmgr.exists(checkpoint_path), f"Checkpoint '{checkpoint_path}' not found."
    with g_pathmgr.open(checkpoint_path, "rb") as f:
        checkpoint = torch.load(f, map_location=map_location, weights_only=weights_only)

    # Extract the state dict
    pre_train_dict = get_state_dict(checkpoint, ckpt_state_dict_keys)
    
    if img_enc_key:
        # Ensure the image encoder key exists
        if not any(img_enc_key in key for key in pre_train_dict.keys()):
            raise KeyError(f"Key '{img_enc_key}' not found in any of the checkpoint state_dict keys.")

        # Extract only the image encoder's state dict
        img_enc_state_dict = {
            key: value for key, value in pre_train_dict.items() if img_enc_key in key and "decoder" not in key
        }
    # If null value is passed, only remove the decoder keys
    else: 
        # Extract only the image encoder's state dict
        img_enc_state_dict = {
            key: value for key, value in pre_train_dict.items() if "decoder" not in key
        }

    # Apply kernels if provided
    if checkpoint_kernels is not None:
        for kernel in checkpoint_kernels:
            img_enc_state_dict = kernel(state_dict=img_enc_state_dict)

    logging.debug(
        "Loaded Only Image Encoder State Dict Post-kernel application %s"
        % str(", ".join(list(img_enc_state_dict.keys())))
    )

    return img_enc_state_dict

def load_dual_checkpoint(
    img_enc_checkpoint_path: str,
    rest_checkpoint_path: str,
    rest_key: str = None,
    img_enc_key: str = "image_encoder",
    img_enc_kernels: List[Callable] = None,
    rest_kernels: List[Callable] = None,
    img_enc_ckpt_state_dict_keys: Tuple[str] = ("state_dict",),
    rest_ckpt_state_dict_keys: Tuple[str] = ("state_dict",),
    map_location: str = "cpu",
    strict: bool = True,
    ignore_missing_keys: List[str] = None,
    ignore_unexpected_keys: List[str] = None,
    fpn_adapation: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Loads weights from two separate checkpoints: one for the image encoder and the other for the rest of the model.

    Args:
        img_enc_checkpoint_path (str): Path to the checkpoint for the image encoder.
        rest_checkpoint_path (str): Path to the checkpoint for the rest of the model.
        img_enc_key (str): The key in the checkpoint corresponding to the image encoder.
        img_enc_kernels (List[Callable]): A list of checkpoint processing kernels for the image encoder.
        rest_kernels (List[Callable]): A list of checkpoint processing kernels for the rest of the model.
        img_enc_ckpt_state_dict_keys (Tuple[str]): Keys containing the image encoder state dict.
        rest_ckpt_state_dict_keys (Tuple[str]): Keys containing the rest of the model state dict.
        map_location (str): Specifies how to remap storage locations.
        strict (bool): Raise if the state_dict has missing state keys.
        ignore_missing_keys (List[str]): Unix pattern of keys to ignore for missing keys.
        ignore_unexpected_keys (List[str]): Unix pattern of keys to ignore for unexpected keys.

    Returns:
        Dict[str, torch.Tensor]: Combined state_dict with weights from both checkpoints.
    """
    # Load the rest of the model weights
    rest_state_dict = load_checkpoint_and_apply_kernels(
        checkpoint_path=rest_checkpoint_path,
        checkpoint_kernels=rest_kernels,
        ckpt_state_dict_keys=rest_ckpt_state_dict_keys,
        map_location=map_location,
    )
    
    # Remove the trunk keys from the state_dict if fpn_adapation else remove the whole encoder
    rest_state_dict = exclude_params_matching_unix_pattern(
        patterns=["image_encoder.*"] if not fpn_adapation else ["image_encoder.trunk.*"],
        state_dict=rest_state_dict,
    )
    
    # Replace rest_state_dict key with rest_key if provided
    if rest_key:
        rest_state_dict = {
            f"{key.replace(rest_key, 'sam_mask_decoder')}": value
            for key, value in rest_state_dict.items()
        }
    
    # Load the image encoder weights
    img_enc_state_dict = load_img_enc_checkpoint_and_apply_kernels(
        checkpoint_path=img_enc_checkpoint_path,
        img_enc_key=img_enc_key,
        checkpoint_kernels=img_enc_kernels,
        ckpt_state_dict_keys=img_enc_ckpt_state_dict_keys,
        map_location=map_location,
        strict=strict,
        ignore_missing_keys=ignore_missing_keys,
        ignore_unexpected_keys=ignore_unexpected_keys,
        weights_only=False,
    )
    
    # If img_enc_key is null, the state_dict is already in right format
    if img_enc_key:
        # Change all keys with "encoder" to "image_encoder.trunk" and ignore keys containing 'mask_token'
        img_enc_state_dict = {
            f"image_encoder.trunk{key.replace(img_enc_key, '')}": value
            for key, value in img_enc_state_dict.items()
            if ('mask_token' not in key) and (img_enc_key)
        }
        
    # Combine the two state_dicts
    combined_state_dict = {**img_enc_state_dict, **rest_state_dict}

    return combined_state_dict


def check_load_state_dict_errors(
    missing_keys,
    unexpected_keys,
    strict: bool,
    ignore_missing_keys: List[str] = None,
    ignore_unexpected_keys: List[str] = None,
):
    if ignore_missing_keys is not None and len(ignore_missing_keys) > 0:
        ignored_keys = unix_pattern_to_parameter_names(
            ignore_missing_keys, missing_keys
        )
        missing_keys = [key for key in missing_keys if key not in ignored_keys]

    if ignore_unexpected_keys is not None and len(ignore_unexpected_keys) > 0:
        ignored_unexpected_keys = unix_pattern_to_parameter_names(
            ignore_unexpected_keys, unexpected_keys
        )
        unexpected_keys = [
            key for key in unexpected_keys if key not in ignored_unexpected_keys
        ]

    err = "State key mismatch."
    if unexpected_keys:
        err += f" Unexpected keys: {unexpected_keys}."
    if missing_keys:
        err += f" Missing keys: {missing_keys}."

    if unexpected_keys or missing_keys:
        logging.warning(err)
        if unexpected_keys or strict:
            raise KeyError(err)


def load_state_dict_into_model(
    state_dict: Dict,
    model: nn.Module,
    strict: bool = True,
    ignore_missing_keys: List[str] = None,
    ignore_unexpected_keys: List[str] = None,
    checkpoint_kernels: List[Callable] = None,
):
    """
    Loads a state dict into the given model.

    Args:
        state_dict: A dictionary containing the model's
            state dict, or a subset if strict is False
        model: Model to load the checkpoint weights into
        strict: raise if the state_dict has missing state keys
        ignore_missing_keys: unix pattern of keys to ignore
    """
    # Apply kernels
    if checkpoint_kernels is not None:
        for f in checkpoint_kernels:
            state_dict = f(state_dict=state_dict)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    # Initialize random values for the missing keys
    for key in missing_keys:
        if key in state_dict:
            continue
        param_shape = model.state_dict()[key].shape
        state_dict[key] = torch.randn(param_shape, dtype=model.state_dict()[key].dtype, device=model.state_dict()[key].device)
    
    logging.debug("Applied random initialization for missing keys:", missing_keys)
    
    check_load_state_dict_errors(
        missing_keys,
        unexpected_keys,
        strict=strict,
        ignore_missing_keys=ignore_missing_keys,
        ignore_unexpected_keys=ignore_unexpected_keys,
    )
    return model

