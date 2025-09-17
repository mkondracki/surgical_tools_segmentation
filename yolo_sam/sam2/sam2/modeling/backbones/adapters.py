import logging
from functools import partial
import math
from typing import List, Tuple, Union

import torch
import torch.nn as nn


from sam2.modeling.backbones.base_block import MultiScaleBlock, MultiScaleAttention


class AdapterBlock(MultiScaleBlock):
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
        scale: float = 0.5,
        attention_type: nn.Module = MultiScaleAttention,
    ):
        super().__init__(
            dim, dim_out, num_heads, mlp_ratio,
            drop_path, norm_layer, q_stride,
            act_layer, window_size, attention_type
        )
        self.Space_Adapter = Adapter(self.dim_out)
        self.MLP_Adapter = Adapter(self.dim_out, skip_connect=False)
        self.scale = scale

    def _process_attention(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(x)
        x = self.Space_Adapter(x)
        return x

    def _process_mlp(self, x: torch.Tensor) -> torch.Tensor:
        xn = self.norm2(x)
        return self.drop_path(self.mlp(xn)) + self.scale * self.MLP_Adapter(xn)
    
    

class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        
    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x