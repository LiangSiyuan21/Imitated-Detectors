# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from fsdet_alter.utils.registry import Registry
from typing import Dict, List
import torch
import torch.nn.functional as F
from torch import nn

from fsdet_alter.layers import ShapeSpec


CHANNEL_GATE_REGISTRY = Registry("CHANNEL_GATE")
CHANNEL_GATE_REGISTRY.__doc__ = """"""

def build_channel_gate(cfg, input_shape):
    name = cfg.MODEL.INSTANCE_SEARCH.CHANNEL_GATE_NAME
    return CHANNEL_GATE_REGISTRY.get(name)(input_shape)
    


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

        
@CHANNEL_GATE_REGISTRY.register()
class ChannelGate(nn.Module):
    """Match Block Net, introduced by the OSOD paper."""
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
            super(ChannelGate, self).__init__()
            self.gate_channels = gate_channels
            self.mlp = nn.Sequential(
                Flatten(),
                nn.Linear(gate_channels, gate_channels // reduction_ratio),
                nn.ReLU(),
                nn.Linear(gate_channels // reduction_ratio, gate_channels)
                )
            self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3)
        return scale      