import logging
from functools import partial
import math
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from sam2.modeling.backbones.lora import Lora
from sam2.modeling.backbones.utils import (
    PatchEmbed,
    window_partition,
    window_unpartition,
)

from sam2.modeling.sam2_utils import DropPath, MLP


def do_pool(x: torch.Tensor, pool: nn.Module, norm: nn.Module = None) -> torch.Tensor:
    if pool is None:
        return x
    # (B, H, W, C) -> (B, C, H, W)
    x = x.permute(0, 3, 1, 2)
    x = pool(x)
    # (B, C, H', W') -> (B, H', W', C)
    x = x.permute(0, 2, 3, 1)
    if norm:
        x = norm(x)

    return x

class MultiScaleAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        q_pool: nn.Module = None,
    ):
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.q_pool = q_pool
        self.qkv = nn.Linear(dim, dim_out * 3)
        self.proj = nn.Linear(dim_out, dim_out)

    def _transform_qkv(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Override this in subclasses to modify q,k,v
        return q, k, v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (B, H * W, 3, nHead, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1)
        # q, k, v with shape (B, H * W, nheads, C)
        q, k, v = torch.unbind(qkv, 2)
        
        # Apply any transformations to q,k,v (e.g. LoRA in subclasses)
        q, k, v = self._transform_qkv(q, k, v)

        # Q pooling (for downsample at stage changes)
        if self.q_pool:
            q = do_pool(q.reshape(B, H, W, -1), self.q_pool)
            H, W = q.shape[1:3]  # downsampled shape
            q = q.reshape(B, H * W, self.num_heads, -1)

        # Torch's SDPA expects [B, nheads, H*W, C] so we transpose
        x = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )
        # Transpose back
        x = x.transpose(1, 2)
        x = x.reshape(B, H, W, -1)

        x = self.proj(x)

        return x
    

class MultiScaleBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        norm_layer: Union[nn.Module, str] = "LayerNorm",
        q_stride: Tuple[int, int] = None,
        act_layer: nn.Module = nn.GELU,
        window_size: int = 0,
        attention_block: nn.Module = MultiScaleAttention,
        **kwargs,
    ):
        super().__init__()

        if isinstance(norm_layer, str):
            norm_layer = partial(getattr(nn, norm_layer), eps=1e-6)

        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)
        self.window_size = window_size

        self.pool, self.q_stride = None, q_stride
        if self.q_stride:
            self.pool = nn.MaxPool2d(
                kernel_size=q_stride, stride=q_stride, ceil_mode=False
            )

        # if attention_type == MultiScaleAttention:
        #     self.attn = attention_type(
        #         dim,
        #         dim_out,
        #         num_heads=num_heads,
        #         q_pool=self.pool,
        #     )
        # elif attention_type == LoraMultiScaleAttention:
        #     self.attn = attention_type(
        #         dim,
        #         dim_out,
        #         num_heads=num_heads,
        #         q_pool=self.pool,
        #         lora_r=lora_r,
        #         lora_scaling=lora_alpha,
        #     )
        
        self.attn = attention_block(
            dim=dim,
            dim_out=dim_out,
            num_heads=num_heads,
            q_pool=self.pool,
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim_out)
        self.mlp = MLP(
            dim_out,
            int(dim_out * mlp_ratio),
            dim_out,
            num_layers=2,
            activation=act_layer,
        )

        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

    def _process_attention(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn(x)

    def _process_mlp(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x  # B, H, W, C
        x = self.norm1(x)

        # Skip connection
        if self.dim != self.dim_out:
            shortcut = do_pool(self.proj(x), self.pool)

        # Window partition
        window_size = self.window_size
        if window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, window_size)

        # Window Attention + Q Pooling (if stage change)
        x = self._process_attention(x)

        if self.q_stride:
            # Shapes have changed due to Q pooling
            window_size = self.window_size // self.q_stride[0]
            H, W = shortcut.shape[1:3]
            pad_h = (window_size - H % window_size) % window_size
            pad_w = (window_size - W % window_size) % window_size
            pad_hw = (H + pad_h, W + pad_w)

        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, window_size, pad_hw, (H, W))

        x = shortcut + self.drop_path(x)
        
        # MLP
        x = x + self._process_mlp(x)
        return x
    
    
    
class LoraMultiScaleAttention(MultiScaleAttention):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        q_pool: nn.Module = None,
        lora_r: int = 4,  # Default LoRA rank
        lora_scaling: float = 1.0,
    ):
        super().__init__(dim, dim_out, num_heads, q_pool)
        
        # LoRA layers
        self.lora_q = Lora(dim_out, lora_r, lora_scaling)
        # self.lora_k = Lora(dim_out, lora_r)
        self.lora_v = Lora(dim_out, lora_r, lora_scaling)

    def _transform_qkv(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Apply LoRA transformations
        """ https://arxiv.org/abs/2106.09685: 
        'With r = 4 and only the query and value projection matrices being adapted' 
        """
        q = q + self.lora_q(q.reshape(-1, self.dim_out)).view(q.shape)
        # k = k + self.lora_k(k.reshape(-1, self.dim)).view(k.shape)
        v = v + self.lora_v(v.reshape(-1, self.dim_out)).view(v.shape)
        return q, k, v