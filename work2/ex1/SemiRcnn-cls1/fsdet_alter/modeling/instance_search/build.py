from fsdet_alter.config.config import CfgNode
from fsdet_alter.utils.registry import Registry

from typing import Dict, List
import torch
import torch.nn.functional as F
from torch import nn

from fsdet_alter.layers import ShapeSpec
from .channel_gate import build_channel_gate


INSTANCE_SEARCH_REGISTRY = Registry("INSTANCE_SEARCH")
INSTANCE_SEARCH_REGISTRY.__doc__ = """

Registry for instance search, which produces instance match from feature maps.

The registered object will be called with `obj(cfg, input_shape)`.
The call should return a `nn.Module` object.
"""

def build_instance_search(cfg, input_shape):
    
    name = cfg.MODEL.INSTANCE_SEARCH.NAME
    return INSTANCE_SEARCH_REGISTRY.get(name)(cfg, input_shape)


@INSTANCE_SEARCH_REGISTRY.register()
class MatchNet(nn.Module):
    """Match Block Net, introduced by the OSOD paper."""

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()

        self.sub_sample = False
        self.in_features = input_shape
        temps = [input_shape[f] for f in self.in_features]
        self.in_channels = [s.channels for s in temps]
        assert len(set(self.in_channels)) == 1, "Each level must have the same channel!"
        self.in_channels = self.in_channels[0]
        self.inter_channels = None

        if self.inter_channels is None:
            self.inter_channels = self.in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1


        conv_nd = nn.Conv2d
        max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        bn = nn.BatchNorm2d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(
            conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            bn(self.in_channels)
        )
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)

        self.Q = nn.Sequential(
            conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            bn(self.in_channels)
        )
        nn.init.constant_(self.Q[1].weight, 0)
        nn.init.constant_(self.Q[1].bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        self.concat_project = nn.Sequential(
            nn.Conv2d(self.inter_channels * 2, 1, 1, 1, 0, bias=False),
            nn.ReLU()
        )
        
        self.ChannelGate = build_channel_gate(cfg ,self.in_channels)
        self.globalAvgPool = nn.AdaptiveAvgPool2d(1)

    def forward(self, detect, aim):
        feature_nums = len(self.in_features)
        detects = [detect[f] for f in self.in_features]
        aims = [aim[f] for f in self.in_features]
        
        batch_sizes, channels, height_ds, width_ds, height_as, width_as = [], [], [], [], [], []
        for x in detects:
            batch_sizes.append(x.shape[0])
            channels.append(x.shape[1])
            height_ds.append(x.shape[2])
            width_ds.append(x.shape[3])
        for x in aims:
            height_as.append(x.shape[2])
            width_as.append(x.shape[3])

        non_dets, act_dets, act_aims, c_weights = [], [], [], []
        #####################################find aim image similar object ####################################################
        for i in range(feature_nums):
            d_x = self.g(detects[i]).view(batch_sizes[i], self.inter_channels, -1)
            d_x = d_x.permute(0, 2, 1).contiguous()

            a_x = self.g(aims[i]).view(batch_sizes[i], self.inter_channels, -1)
            a_x = a_x.permute(0, 2, 1).contiguous()

            theta_x = self.theta(aims[i]).view(batch_sizes[i], self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)

            phi_x = self.phi(detects[i]).view(batch_sizes[i], self.inter_channels, -1)

            

            f = torch.matmul(theta_x, phi_x)

            N = f.size(-1)
            f_div_C = f / N

            f = f.permute(0, 2, 1).contiguous()
            N = f.size(-1)
            fi_div_C = f / N

            non_aim = torch.matmul(f_div_C, d_x)
            non_aim = non_aim.permute(0, 2, 1).contiguous()
            non_aim = non_aim.view(batch_sizes[i], self.inter_channels, height_as[i], width_as[i])
            non_aim = self.W(non_aim)
            non_aim = non_aim + aims[i]

            non_det = torch.matmul(fi_div_C, a_x)
            non_det = non_det.permute(0, 2, 1).contiguous()
            non_det = non_det.view(batch_sizes[i], self.inter_channels, height_ds[i], width_ds[i])
            non_det = self.Q(non_det)
            non_det = non_det + detects[i]

            ##################################### Response in chaneel weight ####################################################

            c_weight = self.ChannelGate(non_aim)
            act_aim = non_aim * c_weight
            act_det = non_det * c_weight

            non_dets.append(non_det)
            act_dets.append(act_det)
            act_aims.append(act_aim)
            c_weights.append(c_weight)

        return non_dets, act_dets, act_aims, c_weights