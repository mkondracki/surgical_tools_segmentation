import logging
from functools import partial
import math
from typing import List, Tuple, Union
import torch.nn.functional as F

import torch
import torch.nn as nn


def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    """
    https://medium.com/data-science/implementing-lora-from-scratch-20f838b046f1:
    Freezes all model parameters except for specific layers and types based on the configuration.
    Parameters in LoRA layers, the finetune head, bias parameters, embeddings, and layer norms 
    can be set as trainable based on class settings.
    """
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False
    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if hasattr(m, 'bias') and \
                m.bias is not None:
                    m.bias.requires_grad = True
    else:
        raise NotImplementedError


class Lora(nn.Module):
    def __init__(self, features, r=4, scaling=1.0):
        super(Lora, self).__init__()
        self.scaling = scaling
        self.linear1 = nn.Linear(features, r, bias=False)
        self.linear2 = nn.Linear(r, features, bias=False)
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.linear1.weight, a=math.sqrt(5))
        nn.init.zeros_(self.linear2.weight)

    def forward(self, x):
        return self.scaling * self.linear2(self.linear1(x))

    


    

