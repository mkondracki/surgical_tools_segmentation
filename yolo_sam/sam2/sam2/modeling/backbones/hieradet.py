# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from functools import partial
import math
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from iopath.common.file_io import g_pathmgr

from sam2.modeling.backbones.utils import (
    PatchEmbed,
    window_partition,
    window_unpartition,
)

from sam2.modeling.sam2_utils import DropPath, MLP

from sam2.modeling.backbones.base_block import MultiScaleBlock, MultiScaleAttention, LoraMultiScaleAttention
from sam2.modeling.backbones.adapters import AdapterBlock
    
    

class Hiera(nn.Module):
    """
    Reference: https://arxiv.org/abs/2306.00989
    """

    def __init__(
        self,
        embed_dim: int = 96,  # initial embed dim
        num_heads: int = 1,  # initial number of heads
        drop_path_rate: float = 0.0,  # stochastic depth
        q_pool: int = 3,  # number of q_pool stages
        q_stride: Tuple[int, int] = (2, 2),  # downsample stride bet. stages
        stages: Tuple[int, ...] = (2, 3, 16, 3),  # blocks per stage
        dim_mul: float = 2.0,  # dim_mul factor at stage shift
        head_mul: float = 2.0,  # head_mul factor at stage shift
        window_pos_embed_bkg_spatial_size: Tuple[int, int] = (14, 14),
        # window size per stage, when not using global att.
        window_spec: Tuple[int, ...] = (
            8,
            4,
            14,
            7,
        ),
        # global attn in these blocks
        global_att_blocks: Tuple[int, ...] = (
            12,
            16,
            20,
        ),
        peft: str = None,
        
        weights_path=None,
        return_interm_layers=True,  # return feats from every stage
    ):
        super().__init__()

        assert len(stages) == len(window_spec)
        self.window_spec = window_spec

        depth = sum(stages)
        self.q_stride = q_stride
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]
        assert 0 <= q_pool <= len(self.stage_ends[:-1])
        self.q_pool_blocks = [x + 1 for x in self.stage_ends[:-1]][:q_pool]
        self.return_interm_layers = return_interm_layers
        self.peft = peft
                

        self.patch_embed = PatchEmbed(
            embed_dim=embed_dim,
        )
        # Which blocks have global att?
        self.global_att_blocks = global_att_blocks

        # Windowed positional embedding (https://arxiv.org/abs/2311.05613)
        self.window_pos_embed_bkg_spatial_size = window_pos_embed_bkg_spatial_size
        self.pos_embed = nn.Parameter(
            torch.zeros(1, embed_dim, *self.window_pos_embed_bkg_spatial_size)
        )
        self.pos_embed_window = nn.Parameter(
            torch.zeros(1, embed_dim, self.window_spec[0], self.window_spec[0])
        )

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        cur_stage = 1
        self.blocks = nn.ModuleList()

        for i in range(depth):
            dim_out = embed_dim
            # lags by a block, so first block of
            # next stage uses an initial window size
            # of previous stage and final window size of current stage
            window_size = self.window_spec[cur_stage - 1]

            if self.global_att_blocks is not None:
                window_size = 0 if i in self.global_att_blocks else window_size

            if i - 1 in self.stage_ends:
                dim_out = int(embed_dim * dim_mul)
                num_heads = int(num_heads * head_mul)
                cur_stage += 1

            # if self.peft == 'adapter':
            #     block = AdapterBlock(
            #         dim=embed_dim,
            #         dim_out=dim_out,
            #         num_heads=num_heads,
            #         drop_path=dpr[i],
            #         q_stride=self.q_stride if i in self.q_pool_blocks else None,
            #         window_size=window_size,
            #         attention_type=MultiScaleAttention
            #     )
                
            # elif self.peft == 'lora':
            #     block= MultiScaleBlock(
            #         dim=embed_dim,
            #         dim_out=dim_out,
            #         num_heads=num_heads,
            #         drop_path=dpr[i],
            #         q_stride=self.q_stride if i in self.q_pool_blocks else None,
            #         window_size=window_size,
            #         attention_type=LoraMultiScaleAttention,
            #         lora_r=4.,
            #         lora_alpha=1.,
            #     )
                
            # else:
            #     block = MultiScaleBlock(
            #         dim=embed_dim,
            #         dim_out=dim_out,
            #         num_heads=num_heads,
            #         drop_path=dpr[i],
            #         q_stride=self.q_stride if i in self.q_pool_blocks else None,
            #         window_size=window_size,
            #         attention_type=MultiScaleAttention
            #     )
            
            block = None
            if callable(self.peft):
                peft_kwargs = dict(
                    dim=embed_dim,
                    dim_out=dim_out,
                    num_heads=num_heads,
                    drop_path=dpr[i],
                    q_stride=self.q_stride if i in self.q_pool_blocks else None,
                    window_size=window_size,
                )
                # If attention_block is a partial, fill in its required args
                if hasattr(self.peft.keywords.get('attention_block', None), 'func'):
                    attn_partial = self.peft.keywords['attention_block']
                    peft_kwargs['attention_block'] = attn_partial
                block = self.peft(**peft_kwargs)
                
            elif self.peft == 'lora':
                block= MultiScaleBlock(
                    dim=embed_dim,
                    dim_out=dim_out,
                    num_heads=num_heads,
                    drop_path=dpr[i],
                    q_stride=self.q_stride if i in self.q_pool_blocks else None,
                    window_size=window_size,
                    attention_block=LoraMultiScaleAttention
                )
                
            else:
                block = MultiScaleBlock(
                    dim=embed_dim,
                    dim_out=dim_out,
                    num_heads=num_heads,
                    drop_path=dpr[i],
                    q_stride=self.q_stride if i in self.q_pool_blocks else None,
                    window_size=window_size,
                    attention_block=MultiScaleAttention
                )
                
            embed_dim = dim_out
            self.blocks.append(block)

        self.channel_list = (
            [self.blocks[i].dim_out for i in self.stage_ends[::-1]]
            if return_interm_layers
            else [self.blocks[-1].dim_out]
        )

        if weights_path is not None:
            with g_pathmgr.open(weights_path, "rb") as f:
                chkpt = torch.load(f, map_location="cpu")
            logging.info("loading Hiera", self.load_state_dict(chkpt, strict=False))

    def _get_pos_embed(self, hw: Tuple[int, int]) -> torch.Tensor:
        h, w = hw
        window_embed = self.pos_embed_window
        pos_embed = F.interpolate(self.pos_embed, size=(h, w), mode="bicubic")
        pos_embed = pos_embed + window_embed.tile(
            [x // y for x, y in zip(pos_embed.shape, window_embed.shape)]
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1)
        return pos_embed

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.patch_embed(x)
        # x: (B, H, W, C)

        # Add pos embed
        x = x + self._get_pos_embed(x.shape[1:3])

        outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if (i == self.stage_ends[-1]) or (
                i in self.stage_ends and self.return_interm_layers
            ):
                feats = x.permute(0, 3, 1, 2)
                outputs.append(feats)

        return outputs

    def get_layer_id(self, layer_name):
        # https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
        num_layers = self.get_num_layers()

        if layer_name.find("rel_pos") != -1:
            return num_layers + 1
        elif layer_name.find("pos_embed") != -1:
            return 0
        elif layer_name.find("patch_embed") != -1:
            return 0
        elif layer_name.find("blocks") != -1:
            return int(layer_name.split("blocks")[1].split(".")[1]) + 1
        else:
            return num_layers + 1

    def get_num_layers(self) -> int:
        return len(self.blocks)

