import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
from degrade_model import get_turbulence_forward_model, loss_physics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MutualLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def constraint_loss_cn2(self, data):
        P = 1013.25  
        eps = 1e-6  
        denom = (data[:, :, 2] * 320.0).clamp(min=eps)
        Cn2_compute = data[:, :, 1] * 14.0 * (79e-6 * P / (denom ** 2)) ** 2 / (8e-12)
        constraint_loss = torch.mean(torch.abs(Cn2_compute - data[:, :, 0]))
        return constraint_loss

    def constraint_loss_energy_dissipation_rate(self, data):
        P = 1013.25 
        gamma = 1.08e22
        x0 = data[:, :, 0].clamp(min=0.0)
        factor = (data[:, :, 2] * 320 / P).clamp(min=1e-6)
        energy_dissipation_rate_compute = gamma * ((x0 * 8e-12) ** 1.5) * (factor ** 3) / 8500
        constraint_loss = torch.mean(torch.abs(energy_dissipation_rate_compute - data[:, :, 3]))
        return constraint_loss

    def constraint_loss_Reynolds_stress(self, data):
        L = 0.1
        x3 = data[:, :, 3].clamp(min=0.0)
        Reynolds_stress_compute = (x3 * 8500 * L) ** (2 / 3) / 90
        constraint_loss = torch.mean(torch.abs(Reynolds_stress_compute - data[:, :, 4]))
        return constraint_loss

        
    def forward(self, restored, cn2_pred, target_img, target_cn2, degraded):

        loss_mit = F.l1_loss(restored, target_img)

        loss_est = F.l1_loss(cn2_pred, target_cn2)
        loss_phys =  self.constraint_loss_cn2(cn2_pred) + self.constraint_loss_energy_dissipation_rate(cn2_pred)
        + self.constraint_loss_Reynolds_stress(cn2_pred)

        physics_params = {
            'T0': 298.15, 'Em': 0.7, 'Trans_air': 0.99,
            'lamda0': 8.0e-6, 'lamda1': 13.0e-6,
            'cam_f': 200e-3, 'cam_F': 1/4, 'cam_lamda': 5000e-9,
            'L': 5000, 'tur_l0': 2e-3, 'tur_L0': 3,
            'tur_delta': 1, 'tur_alpha': 11/3, 'patch_size': 16
        }
        physics_params['cam_D_aperture'] = physics_params['cam_f'] * physics_params['cam_F']
        physics_params['wave_k'] = 2 * 3.14159 / physics_params['cam_lamda']

        forward_model = get_turbulence_forward_model(physics_params, device=device)
        forward_model.eval()

        loss_redeg = loss_physics(
        restored[:,15,:,:,:],       # (B, C, H, W)
        degraded[:,15,:,:,:],    # (B, C, H, W)
        cn2_pred[:,15,:,:,:],          # (B, H//16, W//16, 3)
        forward_model,
        physics_params
    )
        
        total_loss = loss_mit + loss_est + loss_redeg + 0.5 * loss_phys
        
        return {
            'total': total_loss,
            'restoration': loss_mit,
            'estimation': loss_est,
            're_degraded': loss_redeg,
            'physical': loss_phys
        }