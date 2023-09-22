import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
import numpy as np
import random
import collections


class RandomShiftsAug(nn.Module):
    def __init__(self, pad: int = 4):
        super().__init__()
        self.pad = pad
# class RandomShiftsAug(nn.Module):
#     def __init__(self, pad: int = 4):
#         super().__init__()
#         self.pad = pad

#     def forward(self, x: torch.Size([128, 9, 120, 360])): # [512, 9, 84, 420]
#         n, c, h, w = x.size()

#         if h != w:
#             x = x.repeat(1, 1, 360//120, 1) #针对carla场景设定
#         n, c, h, w = x.size()
#         assert h == w
#         padding = tuple([self.pad] * 4)
#         x = F.pad(x, padding, 'replicate')
#         eps = 1.0 / (h + 2 * self.pad)
#         arange = torch.linspace(-1.0 + eps,
#                                 1.0 - eps,
#                                 h + 2 * self.pad,
#                                 device=x.device,
#                                 dtype=x.dtype)[:h]
#         arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
#         base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
#         base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

#         shift = torch.randint(0,
#                               2 * self.pad + 1,
#                               size=(n, 1, 1, 2),
#                               device=x.device,
#                               dtype=x.dtype)
#         shift *= 2.0 / (h + 2 * self.pad)

#         grid = base_grid + shift

#         out = F.grid_sample(x,
#                              grid,
#                              padding_mode='zeros',
#                              align_corners=False)
        
#         out = out[:,:,0:120,:]
#         return out
    def forward(self, x: torch.Size([128, 9, 84, 420])): # [512, 9, 84, 420]
        n, c, h, w = x.size()

        if h != w:
            x = x.repeat(1, 1, 420//84, 1) #针对carla场景设定
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift

        out = F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)
        
        out = out[:,:,0:84,:]
        return out
