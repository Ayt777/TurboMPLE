import torch
import torch.nn as nn
import torch.nn.functional as F
from TASC import Turb_Expert


class MutualPLENetwork(nn.Module):

    def __init__(self, num_levels=3, num_shared_experts=2, 
                 num_task_experts=2, expert_dim=128):
        super().__init__()
        
        self.num_levels = num_levels
        self.expert_dim = expert_dim
        
        self.spatiotemporal_encoder = SpatioTemporalEncoder(
            in_channels=1, 
            out_dim=expert_dim
        )
        
        self.ple_levels = nn.ModuleList([
            MutualPLELevel(
                num_shared_experts=num_shared_experts,
                num_task_experts=num_task_experts,
                expert_dim=expert_dim,
                level=i,
                mutual_weight=0.2 + i * 0.4
            )
            for i in range(num_levels)
        ])
        
        self.restoration_head = RestorationHead(expert_dim)
        self.cn2_head = Cn2EstimationHead(expert_dim)
        
    def forward(self, degraded_sequence, return_features=False):

        B, T, C, H, W = degraded_sequence.shape

        feat = self.spatiotemporal_encoder(degraded_sequence)  
        
        feat_restoration = feat
        feat_cn2 = feat
        
        intermediate_features = []
        for level_idx, ple_level in enumerate(self.ple_levels):
            feat_restoration, feat_cn2 = ple_level(feat_restoration, feat_cn2)
            if return_features:
                intermediate_features.append({
                    'level': level_idx,
                    'feat_restoration': feat_restoration,
                    'feat_cn2': feat_cn2
                })

        restored_sequence = self.restoration_head(feat_restoration, (H, W), degraded_sequence)  # [B, T, 1, H, W]
        cn2_sequence = self.cn2_head(feat_cn2)  # [B, T, 5, H/16, W/16]
        
        if return_features:
            return restored_sequence, cn2_sequence, intermediate_features
        return restored_sequence, cn2_sequence


class SpatioTemporalEncoder(nn.Module):
    def __init__(self, in_channels=1, out_dim=128):
        super().__init__()
        
        self.BLB = BLB(width=128, height=128, channels=1)

        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=(3, 7, 7), 
                      stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )  # stride 2x in spatial
        
        self.conv3d_2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), 
                      stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )  # stride 2x in spatial (total 4x)
        
        self.conv3d_3 = nn.Sequential(
            nn.Conv3d(64, out_dim, kernel_size=(3, 3, 3), 
                      stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(out_dim),
            nn.ReLU(inplace=True)
        )  # no spatial stride
        
        self.temporal_attention = TemporalAttention(out_dim)
        
    def forward(self, x):
        """
        Args:
            x: [B, T, C, H, W]
        Returns:
            feat: [B, T, D, H/4, W/4]
        """
        B, T, C, H, W = x.shape
        
        x = self.BLB(x)
        x = x.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        
        x = self.conv3d_1(x)  # [B, 32, T, H/2, W/2]
        x = self.conv3d_2(x)  # [B, 64, T, H/4, W/4]
        x = self.conv3d_3(x)  # [B, out_dim, T, H/4, W/4]
        
        x = x.permute(0, 2, 1, 3, 4)  # [B, T, out_dim, H/4, W/4]
        x = self.temporal_attention(x)
        
        return x


class TemporalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Conv2d(dim, dim, 1)
        self.key = nn.Conv2d(dim, dim, 1)
        self.value = nn.Conv2d(dim, dim, 1)
        self.scale = dim ** -0.5
        
    def forward(self, x):
        """
        Args:
            x: [B, T, D, H, W]
        Returns:
            out: [B, T, D, H, W]
        """
        B, T, D, H, W = x.shape
        x_reshaped = x.reshape(B * T, D, H, W)
        
        q = self.query(x_reshaped).reshape(B, T, D, H * W)
        k = self.key(x_reshaped).reshape(B, T, D, H * W)
        v = self.value(x_reshaped).reshape(B, T, D, H * W)
        
        attn = torch.einsum('btdh,bsdh->bts', q, k) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = torch.einsum('bts,bsdh->btdh', attn, v)
        out = out.reshape(B, T, D, H, W)
        
        return out + x 


class MutualPLELevel(nn.Module):
    def __init__(self, num_shared_experts, num_task_experts, 
                 expert_dim, level, mutual_weight=0.3):
        super().__init__()
        
        self.level = level
        self.mutual_weight = mutual_weight
        
        self.shared_experts = nn.ModuleList([
            Turb_Expert(hidden_dim=expert_dim, d_state=16, drop_path=0.1) for _ in range(num_task_experts)
            # RDB(in_channels=expert_dim, growthRate=16, num_layer=3, activation='relu')
        ])
        
        self.restoration_experts = nn.ModuleList([
            RestorationExpert(expert_dim) for _ in range(num_task_experts)
        ])
        
        self.cn2_experts = nn.ModuleList([
            Cn2Expert(expert_dim) for _ in range(num_task_experts)
        ])
        
        self.gate_restoration = GateNetwork(
            expert_dim, 
            num_shared_experts + num_task_experts
        )
        self.gate_cn2 = GateNetwork(
            expert_dim, 
            num_shared_experts + num_task_experts
        )
        
        self.mutual_assistance = CrossTaskMutualModule(expert_dim)
        
    def forward(self, feat_restoration, feat_cn2):
        """
        Args:
            feat_restoration: [B, T, D, H, W]
            feat_cn2: [B, T, D, H, W]
        Returns:
            out_restoration: [B, T, D, H, W]
            out_cn2: [B, T, D, H, W]
        """
        B, T, D, H, W = feat_restoration.shape
        
        feat_r = feat_restoration.reshape(B * T, D, H, W)
        feat_c = feat_cn2.reshape(B * T, D, H, W)
        
        shared_outputs_r = [expert(feat_r) for expert in self.shared_experts]
        shared_outputs_c = [expert(feat_c) for expert in self.shared_experts]
        
        restoration_outputs = [expert(feat_r) for expert in self.restoration_experts]
        cn2_outputs = [expert(feat_c) for expert in self.cn2_experts]
        
        restoration_outputs_enhanced, cn2_outputs_enhanced = \
            self.mutual_assistance(
                restoration_outputs, 
                cn2_outputs, 
                self.mutual_weight
            )

        all_restoration = shared_outputs_r + restoration_outputs_enhanced
        # all_restoration = shared_outputs_r + restoration_outputs
        out_r = self.gate_restoration(feat_r, all_restoration)
        
        all_cn2 = shared_outputs_c + cn2_outputs_enhanced
        # all_cn2 = shared_outputs_c + cn2_outputs
        out_c = self.gate_cn2(feat_c, all_cn2)
        
        out_restoration = out_r.reshape(B, T, D, H, W)
        out_cn2 = out_c.reshape(B, T, D, H, W)
        
        return out_restoration, out_cn2


class ExpertModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(dim)
        self.conv2 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out

def actFunc(act, *args, **kwargs):
    act = act.lower()
    if act == 'relu':
        return nn.ReLU()
    elif act == 'relu6':
        return nn.ReLU6()
    elif act == 'leakyrelu':
        return nn.LeakyReLU(0.1)
    elif act == 'prelu':
        return nn.PReLU()
    elif act == 'rrelu':
        return nn.RReLU(0.1, 0.3)
    elif act == 'selu':
        return nn.SELU()
    elif act == 'celu':
        return nn.CELU()
    elif act == 'elu':
        return nn.ELU()
    elif act == 'gelu':
        return nn.GELU()
    elif act == 'tanh':
        return nn.Tanh()
    else:
        raise NotImplementedError


# Dense layer
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=True)


class dense_layer(nn.Module):
    def __init__(self, in_channels, growthRate, activation='relu'):
        super(dense_layer, self).__init__()
        self.conv = conv3x3(in_channels, growthRate)
        self.act = actFunc(activation)

    def forward(self, x):
        out = self.act(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


class CA(nn.Module):
    def __init__(self, channel, down=16):
        super(CA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            conv1x1(channel, channel // down),
            nn.ReLU(inplace=True),
            conv1x1(channel // down, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class RDB(nn.Module):
    def __init__(self, in_channels, growthRate, num_layer, activation='relu'):
        super(RDB, self).__init__()
        in_channels_ = in_channels
        modules = []
        for i in range(num_layer):
            modules.append(dense_layer(in_channels_, growthRate,
                           activation))
            in_channels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv1x1 = conv1x1(in_channels_, in_channels)
        self.CA = CA(in_channels)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv1x1(out)
        out = self.CA(out)
        out += x
        return out

class RestorationExpert(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        self.detail_branch = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, 1, 1),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 2, dim // 2, 3, 1, 1),
            nn.BatchNorm2d(dim // 2)
        )
        
        self.edge_branch = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, 1, 1, dilation=1),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 2, dim // 2, 3, 1, 1),
            nn.BatchNorm2d(dim // 2)
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0),
            nn.BatchNorm2d(dim)
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        """
        Args:
            x: [BT, D, H, W]
        Returns:
            out: [BT, D, H, W]
        """
        identity = x
        
        detail = self.detail_branch(x)
        edge = self.edge_branch(x)
        
        fused = torch.cat([detail, edge], dim=1)
        out = self.fusion(fused)

        out = self.relu(out + identity)
        # print(out.shape)
        
        return out


class Cn2Expert(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        self.context_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 2, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 2, dim // 2, 1, 1, 0)
        )

        self.local_branch = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, 1, 1),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 2, dim // 2, 3, 1, 1),
            nn.BatchNorm2d(dim // 2)
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0),
            nn.BatchNorm2d(dim)
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        """
        Args:
            x: [BT, D, H, W]
        Returns:
            out: [BT, D, H, W]
        """
        B, C, H, W = x.shape
        identity = x
        
        context = self.context_branch(x)  # [BT, D/2, 1, 1]
        context = context.expand(-1, -1, H, W)
        
        local = self.local_branch(x)  # [BT, D/2, H, W]
        
        fused = torch.cat([context, local], dim=1)
        out = self.fusion(fused)
        
        out = self.relu(out + identity)
        
        return out


class CrossTaskMutualModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
    
        self.cross_attn_r2c = CrossAttention(dim)
        
        self.cross_attn_c2r = CrossAttention(dim)
        
        self.gate_r = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.Sigmoid()
        )
        self.gate_c = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, restoration_experts, cn2_experts, mutual_weight):
        """
        Args:
            restoration_experts: List of [BT, D, H, W]
            cn2_experts: List of [BT, D, H, W]
        Returns:
            enhanced_restoration: List of [BT, D, H, W]
            enhanced_cn2: List of [BT, D, H, W]
        """
        enhanced_restoration = []
        enhanced_cn2 = []
        
        for feat_r, feat_c in zip(restoration_experts, cn2_experts):
            r_from_c = self.cross_attn_r2c(feat_r, feat_c)
            gate_r = self.gate_r(torch.cat([feat_r, r_from_c], dim=1))
            feat_r_enhanced = feat_r * (1 - mutual_weight * gate_r) + \
                            r_from_c * (mutual_weight * gate_r)
            
            c_from_r = self.cross_attn_c2r(feat_c, feat_r)
            gate_c = self.gate_c(torch.cat([feat_c, c_from_r], dim=1))
            feat_c_enhanced = feat_c * (1 - mutual_weight * gate_c) + \
                            c_from_r * (mutual_weight * gate_c)
            
            enhanced_restoration.append(feat_r_enhanced)
            enhanced_cn2.append(feat_c_enhanced)
        
        return enhanced_restoration, enhanced_cn2


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.query = nn.Conv2d(dim, dim, 1)
        self.key = nn.Conv2d(dim, dim, 1)
        self.value = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        
    def forward(self, x, context):
        B, C, H, W = x.shape
        
        # Q, K, V
        q = self.query(x).reshape(B, self.num_heads, self.head_dim, H * W)
        k = self.key(context).reshape(B, self.num_heads, self.head_dim, H * W)
        v = self.value(context).reshape(B, self.num_heads, self.head_dim, H * W)
    
        attn = torch.einsum('bhdn,bhdm->bhnm', q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.einsum('bhnm,bhdm->bhdn', attn, v)
        out = out.reshape(B, C, H, W)
        out = self.proj(out)
        
        return out


class GateNetwork(nn.Module):
    def __init__(self, dim, num_experts):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, num_experts),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x, expert_outputs):
        """
        Args:
            x: [BT, D, H, W]
            expert_outputs: List of [BT, D, H, W]
        Returns:
            out: [BT, D, H, W]
        """
        weights = self.gate(x)  # [BT, num_experts]
        out = 0
        for i, expert_out in enumerate(expert_outputs):
            out = out + weights[:, i:i+1, None, None] * expert_out
        
        return out


class RestorationHead(nn.Module):
    def __init__(self, in_dim, up_scale=4):
        super().__init__()
        
        self.up_scale = up_scale

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_dim, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1),  # ×2
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),   # ×2
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 3, 1, 1)
        )
        self.residual_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, target_size=None, degraded_input=None):
        """
        Args:
            x: [B, T, D, H, W] (H=W/4)
            target_size: (H_orig, W_orig)
            degraded_input: [B, T, 1, H_orig, W_orig]
        Returns:
            out: [B, T, 1, H_orig, W_orig]
        """
        B, T, D, H, W = x.shape
        x = x.reshape(B * T, D, H, W)  # [BT, D, H, W]
        feat = self.conv_layers(x)  # [BT, 16, H, W]
        out = self.upsample(feat)   # [BT, 1, H*4, W*4]
        out = out.reshape(B, T, 1, out.shape[-2], out.shape[-1])
        if degraded_input is not None:
            out = out + self.residual_weight * degraded_input
        return out



class Cn2EstimationHead(nn.Module):
    def __init__(self, in_dim, out_dim=5, block_channels=64, down1_channels=32, down2_channels=16):
        super().__init__()
        self.block_channels = block_channels
        self.down1_channels = down1_channels
        self.down2_channels = down2_channels

        self.block1 = nn.Sequential(
            nn.Conv2d(in_dim, block_channels, 3, 1, 1),
            nn.BatchNorm2d(block_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(block_channels, block_channels, 3, 1, 1),
            nn.BatchNorm2d(block_channels)
        )

        self.shortcut = nn.Conv2d(in_dim, block_channels, 1) if in_dim != block_channels else nn.Identity()

        self.down1 = nn.Conv2d(block_channels, down1_channels, 3, stride=2, padding=1)
        self.down2 = nn.Conv2d(down1_channels, down2_channels, 3, stride=2, padding=1)

        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(down2_channels, down2_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(down2_channels // 4, down2_channels, 1),
            nn.Sigmoid()
        )

        self.out_layer = nn.Conv2d(down2_channels, out_dim, 3, 1, 1)

    def forward(self, x):
        if x.ndim == 5:
            B, T, C, H, W = x.shape
            x = x.reshape(B * T, C, H, W)
        else:
            B = x.shape[0]

        res = self.block1(x)
        x = F.relu(res + self.shortcut(x))

        x = F.relu(self.down1(x))
        x = F.relu(self.down2(x))

        ca = self.ca(x)
        x = x * ca

        out = torch.sigmoid(self.out_layer(x))

        if out.ndim == 4 and 'T' in locals():
            out = out.reshape(B, T, -1, out.shape[2], out.shape[3])

        return out


import torch.nn.functional as F

class BLB(nn.Module):
    def __init__(self, width=128, height=128, channels=1, scale_range=0.05):
        super().__init__()
        self.channels = channels
        self.scale_range = scale_range

        self.coeff_unconstrained = nn.ParameterDict({
            'alpha1': nn.Parameter(torch.zeros(1, channels, height, width)),
            'alpha2': nn.Parameter(torch.zeros(1, channels, height, width)),
            'distance': nn.Parameter(torch.zeros(1, channels, height, width))
        })

        self.scale_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh()
        )

    def forward(self, x):
        """
        x: [N, T, C, H, W]
        Returns:
            output: [N, T, C, H, W]
        """
        N, T, C, H, W = x.shape

        alpha1 = F.interpolate(self.coeff_unconstrained['alpha1'], size=(H, W), mode='bilinear', align_corners=False)
        alpha2 = F.interpolate(self.coeff_unconstrained['alpha2'], size=(H, W), mode='bilinear', align_corners=False)
        distance = F.interpolate(self.coeff_unconstrained['distance'], size=(H, W), mode='bilinear', align_corners=False)
        alpha1 = 0.025 * torch.sigmoid(alpha1)
        alpha2 = 0.025 * torch.sigmoid(alpha2)
        distance = 1.0 + torch.sigmoid(distance)
        alpha_total = alpha1 + alpha2
        time_indices = torch.linspace(0, 1, T, device=x.device).unsqueeze(1)  # [T, 1]
        scale_factor = self.scale_mlp(time_indices).squeeze(-1)  # [T]
        scale_factor = 1 + self.scale_range * (scale_factor - scale_factor.mean())

        alpha_modulated = alpha_total[None, :, :, :] * scale_factor[:, None, None, None]
        attenuation = torch.exp(alpha_modulated * distance)

        output = x * attenuation[None, :, :, :, :].squeeze(0)
        return output
