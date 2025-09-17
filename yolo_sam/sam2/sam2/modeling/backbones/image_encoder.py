# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

class ImageEncoder(nn.Module):
    def __init__(
        self,
        trunk: nn.Module,
        neck: nn.Module,
        scalp: int = 0,
        fpn_adaptation: bool = False,
    ):
        super().__init__()
        self.trunk = trunk
        self.neck = neck
        self.scalp = scalp
        self.fpn_adaptation = fpn_adaptation

        assert ((
            self.trunk.channel_list == self.neck.backbone_channel_list
        ) or fpn_adaptation), f"Channel dims of trunk and neck do not match. Trunk: {self.trunk.channel_list}, neck: {self.neck.backbone_channel_list}"
        
        if fpn_adaptation:
            # Adapt the trunk features to the neck features
            self.fpn_adaptation_conv = FpnAdaptationConv(
                input_channels=self.trunk.channel_list,
                output_channels=self.neck.backbone_channel_list,
            )
            
            
    def forward(self, sample: torch.Tensor):
        # Forward through backbone
        tr_out = self.trunk(sample)
        embedding_1, embedding_2, embedding_3, embedding_4 = tr_out
        features = {
            "embedding_1": embedding_1,
            "embedding_2": embedding_2,
            "embedding_3": embedding_3,
            "embedding_4": embedding_4,
        }
        
        # Apply FpnAdaptation if enabled
        if self.fpn_adaptation:
            tr_out = self.fpn_adaptation_conv([embedding_4, embedding_3, embedding_2, embedding_1])
            tr_out = tr_out[::-1]  # Reverse the order of features
            
        # # Save the features to a file
        # save_path = f"/home/samba/320281459/code/sam2/sam2/modeling/backbones/feature_comparison/features_{self.trunk._get_name()}.pt"
        # torch.save(features, save_path)
        
        # Forward through neck
        features, pos = self.neck(tr_out)
        if self.scalp > 0:
            # Discard the lowest resolution features
            features, pos = features[: -self.scalp], pos[: -self.scalp]

        src = features[-1]
        output = {
            "vision_features": src,
            "vision_pos_enc": pos,
            "backbone_fpn": features,
        }
        return output


class FpnNeck(nn.Module):
    """
    A modified variant of Feature Pyramid Network (FPN) neck
    (we remove output conv and also do bicubic interpolation similar to ViT
    pos embed interpolation)
    """

    def __init__(
        self,
        position_encoding: nn.Module,
        d_model: int,
        backbone_channel_list: List[int],
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        fpn_interp_model: str = "bilinear",
        fuse_type: str = "sum",
        fpn_top_down_levels: Optional[List[int]] = None,
    ):
        """Initialize the neck
        :param trunk: the backbone
        :param position_encoding: the positional encoding to use
        :param d_model: the dimension of the model
        :param neck_norm: the normalization to use
        """
        super().__init__()
        self.position_encoding = position_encoding
        self.convs = nn.ModuleList()
        self.backbone_channel_list = backbone_channel_list
        self.d_model = d_model
        for dim in backbone_channel_list:
            current = nn.Sequential()
            current.add_module(
                "conv",
                nn.Conv2d(
                    in_channels=dim,
                    out_channels=d_model,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
            )

            self.convs.append(current)
        self.fpn_interp_model = fpn_interp_model
        assert fuse_type in ["sum", "avg"]
        self.fuse_type = fuse_type

        # levels to have top-down features in its outputs
        # e.g. if fpn_top_down_levels is [2, 3], then only outputs of level 2 and 3
        # have top-down propagation, while outputs of level 0 and level 1 have only
        # lateral features from the same backbone level.
        if fpn_top_down_levels is None:
            # default is to have top-down features on all levels
            fpn_top_down_levels = range(len(self.convs))
        self.fpn_top_down_levels = list(fpn_top_down_levels)

    def forward(self, xs: List[torch.Tensor]):

        out = [None] * len(self.convs)
        pos = [None] * len(self.convs)
        assert len(xs) == len(self.convs)
        # fpn forward pass
        # see https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/fpn.py
        prev_features = None
        # forward in top-down order (from low to high resolution)
        n = len(self.convs) - 1
        for i in range(n, -1, -1):
            x = xs[i]
            lateral_features = self.convs[n - i](x)
            if i in self.fpn_top_down_levels and prev_features is not None:
                top_down_features = F.interpolate(
                    prev_features.to(dtype=torch.float32),
                    scale_factor=2.0,
                    mode=self.fpn_interp_model,
                    align_corners=(
                        None if self.fpn_interp_model == "nearest" else False
                    ),
                    antialias=False,
                )
                prev_features = lateral_features + top_down_features
                if self.fuse_type == "avg":
                    prev_features /= 2
            else:
                prev_features = lateral_features
            x_out = prev_features
            out[i] = x_out
            pos[i] = self.position_encoding(x_out).to(x_out.dtype)

        return out, pos
    
    
class FpnAdaptationConv(nn.Module):
    def __init__(self, input_channels: List[int], output_channels: List[int]):
        """
        Initialize the FpnAdaptationConv layer.

        :param input_channels: List of input channel sizes for each embedding.
        :param output_channels: List of output channel sizes for each embedding.
        """
        super().__init__()
        assert len(input_channels) == len(output_channels), \
            "Input and output channel lists must have the same length."
        
        # Create a convolutional layer for each embedding
        self.adaptation_layers = nn.ModuleList([
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=1, padding=0)
            for in_ch, out_ch in zip(input_channels, output_channels)
        ])
        

    def forward(self, embeddings: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass for adapting the embeddings.

        :param embeddings: List of input embeddings (torch.Tensors).
        :return: List of adapted embeddings.
        """
        adapted_embeddings = [
            layer(embedding) for layer, embedding in zip(self.adaptation_layers, embeddings)
        ]
        return adapted_embeddings

class FpnResidualAdaptationConv(nn.Module):
    def __init__(self, input_channels: List[int], output_channels: List[int]):
        """
        Initialize the FpnAdaptationConv layer.

        :param input_channels: List of input channel sizes for each embedding.
        :param output_channels: List of output channel sizes for each embedding.
        """
        super().__init__()
        assert len(input_channels) == len(output_channels), \
            "Input and output channel lists must have the same length."
        
        # Create a convolutional layer for each embedding
        self.adaptation_layers = nn.ModuleList([
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=1, padding=0)
            for in_ch, out_ch in zip(input_channels, output_channels)
        ])
        self.residual_layers = nn.ModuleList([
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=1, padding=0)
            for in_ch, out_ch in zip(input_channels, output_channels)
        ])
        
    def forward(self, embeddings: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass for adapting the embeddings.
        :param embeddings: List of input embeddings (torch.Tensors).
        :return: List of adapted embeddings.
        """
        adapted_embeddings = [
            layer(embedding) \
                # + residual_layer(embedding)
            for layer, residual_layer, embedding in zip(self.adaptation_layers, self.residual_layers, embeddings)
        ]
        return adapted_embeddings