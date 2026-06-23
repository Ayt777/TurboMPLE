import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import DropPath
from typing import Callable


def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    batch_size, height, width, num_channels = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batch_size, height, width, groups, channels_per_group)
    x = torch.transpose(x, 3, 4).contiguous()
    x = x.view(batch_size, height, width, -1)
    return x


class TurbulenceAwareConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        
        self.turb_estimator = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 3, 1, 1, groups=channels // 4),  # depthwise
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 1),
            nn.Sigmoid()
        )
        
        self.adaptive_modulator = nn.Sequential(
            nn.Conv2d(channels + 1, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            out: [B, C, H, W]
            turb_map: [B, 1, H, W] 
        """
        turb_map = self.turb_estimator(x)  # [B, 1, H, W]
        combined = torch.cat([x, turb_map], dim=1)  # [B, C+1, H, W]
        modulation_weights = self.adaptive_modulator(combined)  # [B, C, H, W]
        out = x * modulation_weights
        return out, turb_map


class Turb_Expert(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        **kwargs,
    ):
        super().__init__()
        
        self.ln_1 = norm_layer(hidden_dim // 2)
        
        from ss2d import SS2D
        self.self_attention = SS2D(
            d_model=hidden_dim // 2, 
            dropout=attn_drop_rate, 
            d_state=d_state, 
            **kwargs
        )
        self.drop_path = DropPath(drop_path)
        
        self.turbulence_aware = TurbulenceAwareConv(hidden_dim // 2)
        
        self.conv33conv33conv11 = nn.Sequential(
            nn.BatchNorm2d(hidden_dim // 2),
            nn.Conv2d(hidden_dim // 2, hidden_dim // 2, 3, 1, 1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim // 2, 3, 1, 1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim // 2, 1, 1),
            nn.ReLU()
        )
        
    def forward(self, input: torch.Tensor, return_turb_map=False):
        input = input.permute(0, 2, 3, 1)
        input_left, input_right = input.chunk(2, dim=-1)
        
        x_right = self.drop_path(self.self_attention(self.ln_1(input_right)))
        input_left = input_left.permute(0, 3, 1, 2).contiguous()
        input_left_modulated, turb_map = self.turbulence_aware(input_left)
        conv_out = self.conv33conv33conv11(input_left_modulated)
        conv_out = conv_out.permute(0, 2, 3, 1).contiguous()

        output = torch.cat((conv_out, x_right), dim=-1)
        output = channel_shuffle(output, groups=2)
        output = (output + input).permute(0, 3, 1, 2)
        
        if return_turb_map:
            return output, turb_map
        return output