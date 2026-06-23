import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from scipy.special import gamma as scipy_gamma
from scipy.integrate import quad
from scipy.io import loadmat
import os


def Mt_numpy(lamda0, lamda1, T):
    C1 = 3.7418e-16  
    C2 = 1.4388e-2   
    eps = 1e-12       
    
    T = max(T, eps)

    def integrand(lamda):
        lamda = max(lamda, eps)  
        x = C2 / (lamda * T)
        x = np.clip(x, 0, 700)
        return C1 / (lamda**5) / (np.expm1(x) + eps) 

    Mt_T, _ = quad(integrand, lamda0, lamda1, limit=200)
    return Mt_T

def Gray_change_numpy(T0, T1, Em, Trans_air, lamda0, lamda1):
    Mt_T0 = Mt_numpy(lamda0, lamda1, T0)
    Mt_T1 = Mt_numpy(lamda0, lamda1, T1)
    
    temp1 = Trans_air * Em
    temp2 = (1 - Em * Trans_air) * Mt_T1 / Mt_T0
    gray_rate = temp1 + temp2
    
    return gray_rate

def Calculate_MTF_numpy(l0, L0, alpha, u, D, L, Cn2, k):
    A_alpha = scipy_gamma(alpha - 1) / (4 * np.pi**2) * np.cos(alpha * np.pi / 2)
    
    kL = 4 * np.pi / L0
    c_alpha = np.pi * A_alpha * (scipy_gamma(-alpha/2 + 3/2) * (3 - alpha) / 3)
    c_alpha = c_alpha ** (1 / (alpha - 5))
    kl = c_alpha / l0
    
    p = u * D
    z = -p**2 * kl**2 / 4
    
    temp1 = -(1 - alpha/2) * z * (1 - z * (2/(alpha-2)/scipy_gamma(alpha/2))**(2/(alpha-4)))**(alpha/2 - 2)
    d1_pl = 0.5 * scipy_gamma(-alpha/2 + 1) * temp1
    
    D1_pl = 8 * np.pi**2 * L * Cn2 * k**2 * A_alpha * kl**(2 - alpha) * d1_pl
    
    k_temp = 1/kl**2 + 1/kL**2
    z1 = -p**2 / k_temp / 4
    
    temp2 = -(1 - alpha/2) * z1 * (1 - z1 * (2/(alpha-2)/scipy_gamma(alpha/2))**(2/(alpha-4)))**(alpha/2 - 2)
    d2_pl = 0.5 * k_temp**(alpha/2 - 1) * scipy_gamma(-alpha/2 + 1) * temp2
    
    D2_pl = 8 * np.pi**2 * L * Cn2 * k**2 * A_alpha * d2_pl
    
    D_pl = D1_pl - D2_pl
    MTF_pl = np.exp(-0.5 * D_pl)
    
    return MTF_pl

def otf2psf_numpy(otf, out_size=None):
    psf = np.fft.ifftn(otf)
    psf = np.real(psf)
    
    for axis, axis_size in enumerate(psf.shape):
        psf = np.roll(psf, -axis_size // 2, axis=axis)
    
    if out_size is not None:
        slices = tuple(slice(0, s) for s in out_size)
        psf = psf[slices]
    
    return psf

def ift2_numpy(G, delta_f):
    return np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(G))) * G.shape[0] * G.shape[1] * delta_f**2

def Cal_Random_Matrix_numpy(rows, cols, delta):
    del_x = 1 / (rows * delta)
    del_y = 1 / (cols * delta)
    
    f = np.zeros((rows, cols))
    for indx in range(rows):
        for indy in range(cols):
            f[indx, indy] = np.sqrt((indx - rows/2)**2 * del_x**2 + (indy - cols/2)**2 * del_y**2)
    
    PSD = (f + 0.001)**(-3)
    PSD[rows//2, cols//2] = 0
    
    temp = np.random.randn(rows, cols) + 1j * np.random.randn(rows, cols)
    cn = 2 * np.pi * temp * np.sqrt(PSD) * np.sqrt(del_x * del_y)
    
    phz_hi = ift2_numpy(cn, 1)
    Random_Matrix = np.real(phz_hi)
    
    return Random_Matrix


class DifferentiableGrayShift(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, img, gray_coeff_map):
        if img.dim() == 3:
            img = img.unsqueeze(1)
        if gray_coeff_map.dim() == 3:
            gray_coeff_map = gray_coeff_map.unsqueeze(1)
        
        shifted = img * gray_coeff_map
        return torch.clamp(shifted, 0, 255)


class DifferentiableWarp(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, img, displacement_x, displacement_y):
        B, C, H, W = img.shape
        
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=img.device),
            torch.linspace(-1, 1, W, device=img.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)
        
        displacement_x_norm = 2 * displacement_x / (W - 1)
        displacement_y_norm = 2 * displacement_y / (H - 1)
        
        grid[..., 0] = grid[..., 0] - displacement_x_norm
        grid[..., 1] = grid[..., 1] - displacement_y_norm
        
        warped = F.grid_sample(
            img, 
            grid, 
            mode='bilinear', 
            padding_mode='border',
            align_corners=True
        )
        
        return warped


class DifferentiableSpatiallyVaryingBlur(nn.Module):
    def __init__(self, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
    
    def forward(self, img, psf_blocks):
        B, C, H, W = img.shape
        assert C == 1, "single channel"
        
        n_blocks_h, n_blocks_w = psf_blocks.shape[1], psf_blocks.shape[2]
        psf_h, psf_w = psf_blocks.shape[3], psf_blocks.shape[4]
        
        pad_h, pad_w = psf_h // 2, psf_w // 2
        img_padded = F.pad(img, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
        
        blurred = torch.zeros_like(img)
        
        for bi in range(n_blocks_h):
            for bj in range(n_blocks_w):
                psf = psf_blocks[:, bi, bj, :, :]
                psf = psf / psf.sum(dim=(1, 2), keepdim=True)
                
                h_start = bi * self.patch_size
                h_end = min((bi + 1) * self.patch_size, H)
                w_start = bj * self.patch_size
                w_end = min((bj + 1) * self.patch_size, W)
                
                img_region = img_padded[:, :, h_start:h_end+2*pad_h, w_start:w_end+2*pad_w]
                
                for b in range(B):
                    kernel = psf[b, :, :].view(1, 1, psf_h, psf_w)  # (1, 1, psf_h, psf_w)
                    region = img_region[b, :, :, :].unsqueeze(0)  # (1, 1, region_h, region_w)
                    
                    blurred_region = F.conv2d(
                        input=region,
                        weight=kernel,
                        bias=None,
                        stride=(1, 1),
                        padding=(0, 0),
                        dilation=(1, 1),
                        groups=1
                    )
                    
                    blurred[b, :, h_start:h_end, w_start:w_end] = \
                        blurred_region[0, :, :h_end-h_start, :w_end-w_start]
        
        return blurred


class DifferentiableTurbulenceModel(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.gray_shift = DifferentiableGrayShift()
        self.warp = DifferentiableWarp()
        self.blur = DifferentiableSpatiallyVaryingBlur(patch_size=params['patch_size'])
    
    def forward(self, clean_img, gray_coeff_map, displacement_x, displacement_y, psf_blocks):
        img_shifted = self.gray_shift(clean_img, gray_coeff_map)
        img_warped = self.warp(img_shifted, displacement_x, displacement_y)
        img_blurred = self.blur(img_warped, psf_blocks)
        
        return img_blurred


def precompute_turbulence_params(Turbu_Mat, frames, params):
    N_frames = Turbu_Mat.shape[3]
    rows, cols = frames[0].shape
    patch_size = params['patch_size']
    
    image_key = np.zeros((rows, cols, 2), dtype=int)
    for i in range(rows):
        for j in range(cols):
            image_key[i, j, 0] = i // patch_size
            image_key[i, j, 1] = j // patch_size
    
    gray_coeff_maps = np.zeros((N_frames, rows, cols))
    for index in range(N_frames):
        gray_mat_block = np.zeros((rows // patch_size, cols // patch_size))
        for block_i in range(rows // patch_size):
            for block_j in range(cols // patch_size):
                T_gray = Turbu_Mat[block_i, block_j, 1, index]
                gray_mat_block[block_i, block_j] = Gray_change_numpy(
                    params['T0'], T_gray, params['Em'], 
                    params['Trans_air'], params['lamda0'], params['lamda1']
                )
        
        for i in range(rows):
            for j in range(cols):
                gray_coeff_maps[index, i, j] = gray_mat_block[image_key[i, j, 0], image_key[i, j, 1]]
    
    Delta_x_matrix = np.zeros((rows, cols, N_frames))
    Delta_y_matrix = np.zeros((rows, cols, N_frames))
    
    Delta_x_use, Delta_y_use = None, None
    for index in range(N_frames):
        Rand_Mat_x = Cal_Random_Matrix_numpy(rows, cols, params['tur_delta'])
        Rand_Mat_y = Cal_Random_Matrix_numpy(rows, cols, params['tur_delta'])
        
        if index == 0:
            Delta_x_use = Rand_Mat_x * 0.5
            Delta_y_use = Rand_Mat_y * 0.5
        else:
            Delta_x_use = (Delta_x_use * 2 + Rand_Mat_x) / 3
            Delta_y_use = (Delta_y_use * 2 + Rand_Mat_y) / 3
        
        Delta_x_matrix[:, :, index] = Delta_x_use
        Delta_y_matrix[:, :, index] = Delta_y_use
    
    Delta_x_matrix = (Delta_x_matrix - np.mean(Delta_x_matrix)) / np.std(Delta_x_matrix)
    Delta_y_matrix = (Delta_y_matrix - np.mean(Delta_y_matrix)) / np.std(Delta_y_matrix)
    
    A_alpha = scipy_gamma(params['tur_alpha'] - 1) / (4 * np.pi**2) * np.cos(params['tur_alpha'] * np.pi / 2)
    delta1 = params['tur_delta'] * params['cam_f'] / params['L']
    Delta_theta = delta1 / params['cam_f']
    
    for index in range(N_frames):
        AOA_Mat = np.zeros((rows, cols))
        for i in range(rows):
            for j in range(cols):
                Cn2_AOA = Turbu_Mat[image_key[i, j, 0], image_key[i, j, 1], 0, index]
                AOA_variance = np.pi**2 * A_alpha * params['L'] * Cn2_AOA * \
                               scipy_gamma(2 - params['tur_alpha']/2) / \
                               (0.25 * params['cam_D_aperture']**2 / 4)**(2 - params['tur_alpha']/2)
                AOA_Mat[i, j] = AOA_variance
        
        Delta_x_matrix[:, :, index] *= np.sqrt(AOA_Mat) / Delta_theta
        Delta_y_matrix[:, :, index] *= np.sqrt(AOA_Mat) / Delta_theta
    
    n_blocks_h = rows // patch_size
    n_blocks_w = cols // patch_size
    psf_size = 25
    psf_blocks_all = np.zeros((N_frames, n_blocks_h, n_blocks_w, psf_size, psf_size))
    
    for index in range(N_frames):
        MTF = np.ones((32, 32))
        u_max = np.sqrt(32**2 / 4 + 32**2 / 4)
        
        for Cn2_i in range(n_blocks_h):
            for Cn2_j in range(n_blocks_w):
                Cn2 = Turbu_Mat[Cn2_i, Cn2_j, 0, index]
                
                for u_i in range(32):
                    for u_j in range(32):
                        u = np.sqrt((32/2 - u_i + 1)**2 / 4 + (32/2 - u_j + 1)**2 / 4) / u_max
                        MTF[u_i, u_j] = Calculate_MTF_numpy(
                            params['tur_l0'], params['tur_L0'], params['tur_alpha'], 
                            u, params['cam_D_aperture'], params['L'], Cn2, params['wave_k']
                        )
                
                psf = np.abs(otf2psf_numpy(MTF, out_size=(psf_size, psf_size)))
                psf_blocks_all[index, Cn2_i, Cn2_j, :, :] = psf
    
    return {
        'gray_coeff_maps': gray_coeff_maps,
        'displacement_x': Delta_x_matrix,
        'displacement_y': Delta_y_matrix,
        'psf_blocks': psf_blocks_all,
        'frames': np.array([f for f in frames])
    }


def simulate_turbulence_video_differentiable(
    clean_video_path, 
    cn2_mat_path, 
    output_video_path,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    params = {
        'T0': 298.15,
        'Em': 0.7,
        'Trans_air': 0.99,
        'lamda0': 8.0e-6,
        'lamda1': 13.0e-6,
        'cam_f': 200e-3,
        'cam_F': 1/4,
        'cam_lamda': 5000e-9,
        'L': 5000,
        'tur_l0': 2e-3,
        'tur_L0': 3,
        'tur_delta': 1,
        'tur_alpha': 11/3,
        'patch_size': 16
    }
    params['cam_D_aperture'] = params['cam_f'] * params['cam_F']
    params['wave_k'] = 2 * np.pi / params['cam_lamda']
    
    cap = cv2.VideoCapture(clean_video_path)
    if not cap.isOpened():
        raise ValueError(f"cannot open: {clean_video_path}")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    N_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rows = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cols = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
        frames.append(gray)
    cap.release()
    
    mat_data = loadmat(cn2_mat_path)
    Turbu_Mat = mat_data['Turbu_Mat']
    
    precomputed = precompute_turbulence_params(Turbu_Mat, frames, params)
    
    model = DifferentiableTurbulenceModel(params).to(device)
    model.eval()

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 10, (cols, rows), True)
    
    with torch.no_grad():
        for index in range(N_frames):
            clean_img = torch.from_numpy(precomputed['frames'][index]).float().unsqueeze(0).unsqueeze(0).to(device)
            gray_coeff = torch.from_numpy(precomputed['gray_coeff_maps'][index]).float().unsqueeze(0).unsqueeze(0).to(device)
            disp_x = torch.from_numpy(precomputed['displacement_x'][:, :, index]).float().unsqueeze(0).to(device)
            disp_y = torch.from_numpy(precomputed['displacement_y'][:, :, index]).float().unsqueeze(0).to(device)
            psf = torch.from_numpy(precomputed['psf_blocks'][index]).float().unsqueeze(0).to(device)
            
            turbulent_img = model(clean_img, gray_coeff, disp_x, disp_y, psf)

            img_out = turbulent_img[0, 0].cpu().numpy()
            img_out = np.clip(img_out, 0, 255).astype(np.uint8)
            img_out_color = cv2.cvtColor(img_out, cv2.COLOR_GRAY2BGR)
            out.write(img_out_color)
    
    out.release()
    
    return model, precomputed


def get_turbulence_forward_model(params=None, device='cuda'):
    if params is None:
        params = {
            'T0': 298.15,
            'Em': 0.7,
            'Trans_air': 0.99,
            'lamda0': 8.0e-6,
            'lamda1': 13.0e-6,
            'cam_f': 200e-3,
            'cam_F': 1/4,
            'cam_lamda': 5000e-9,
            'L': 5000,
            'tur_l0': 2e-3,
            'tur_L0': 3,
            'tur_delta': 1,
            'tur_alpha': 11/3,
            'patch_size': 16
        }
        params['cam_D_aperture'] = params['cam_f'] * params['cam_F']
        params['wave_k'] = 2 * np.pi / params['cam_lamda']
    
    model = DifferentiableTurbulenceModel(params).to(device)
    return model


def prepare_turbulence_params_from_cn2(Cn2_field, img_shape, params, device='cuda'):
    if torch.is_tensor(Cn2_field):
        Cn2_field = Cn2_field.cpu().detach().numpy()
    
    rows, cols = img_shape
    patch_size = params['patch_size']
    image_key = np.zeros((rows, cols, 2), dtype=int)
    for i in range(rows):
        for j in range(cols):
            image_key[i, j, 0] = i // patch_size
            image_key[i, j, 1] = j // patch_size
    
    gray_mat_block = np.zeros((rows // patch_size, cols // patch_size))
    for block_i in range(rows // patch_size):
        for block_j in range(cols // patch_size):
            T_gray = Cn2_field[1, block_i, block_j]
            gray_mat_block[block_i, block_j] = Gray_change_numpy(
                params['T0'], T_gray, params['Em'], 
                params['Trans_air'], params['lamda0'], params['lamda1']
            )
    
    gray_coeff_map = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            gray_coeff_map[i, j] = gray_mat_block[image_key[i, j, 0], image_key[i, j, 1]]
    
    Rand_Mat_x = Cal_Random_Matrix_numpy(rows, cols, params['tur_delta'])
    Rand_Mat_y = Cal_Random_Matrix_numpy(rows, cols, params['tur_delta'])
    
    Delta_x = Rand_Mat_x * 0.5
    Delta_y = Rand_Mat_y * 0.5
    
    Delta_x = (Delta_x - np.mean(Delta_x)) / (np.std(Delta_x) + 1e-8)
    Delta_y = (Delta_y - np.mean(Delta_y)) / (np.std(Delta_y) + 1e-8)
    
    A_alpha = scipy_gamma(params['tur_alpha'] - 1) / (4 * np.pi**2) * np.cos(params['tur_alpha'] * np.pi / 2)
    delta1 = params['tur_delta'] * params['cam_f'] / params['L']
    Delta_theta = delta1 / params['cam_f']
    
    AOA_Mat = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            Cn2_AOA = Cn2_field[0, image_key[i, j, 0], image_key[i, j, 1]]
            AOA_variance = np.pi**2 * A_alpha * params['L'] * Cn2_AOA * \
                           scipy_gamma(2 - params['tur_alpha']/2) / \
                           (0.25 * params['cam_D_aperture']**2 / 4)**(2 - params['tur_alpha']/2)
            AOA_Mat[i, j] = AOA_variance
    
    Delta_x *= np.sqrt(np.abs(AOA_Mat)) / Delta_theta
    Delta_y *= np.sqrt(np.abs(AOA_Mat)) / Delta_theta
    n_blocks_h = rows // patch_size
    n_blocks_w = cols // patch_size
    psf_size = 25
    psf_blocks = np.zeros((n_blocks_h, n_blocks_w, psf_size, psf_size))
    
    MTF = np.ones((32, 32))
    u_max = np.sqrt(32**2 / 4 + 32**2 / 4)
    
    for Cn2_i in range(n_blocks_h):
        for Cn2_j in range(n_blocks_w):
            Cn2 = Cn2_field[0, Cn2_i, Cn2_j]
            
            for u_i in range(32):
                for u_j in range(32):
                    u = np.sqrt((32/2 - u_i + 1)**2 / 4 + (32/2 - u_j + 1)**2 / 4) / u_max
                    MTF[u_i, u_j] = Calculate_MTF_numpy(
                        params['tur_l0'], params['tur_L0'], params['tur_alpha'], 
                        u, params['cam_D_aperture'], params['L'], Cn2, params['wave_k']
                    )
            
            psf = np.abs(otf2psf_numpy(MTF, out_size=(psf_size, psf_size)))
            psf_blocks[Cn2_i, Cn2_j, :, :] = psf
    
    return {
        'gray_coeff': torch.from_numpy(gray_coeff_map).float().unsqueeze(0).unsqueeze(0).to(device),  # (1, 1, H, W)
        'disp_x': torch.from_numpy(Delta_x).float().unsqueeze(0).to(device),  # (1, H, W)
        'disp_y': torch.from_numpy(Delta_y).float().unsqueeze(0).to(device),  # (1, H, W)
        'psf': torch.from_numpy(psf_blocks).float().unsqueeze(0).to(device)  # (1, n_h, n_w, psf_h, psf_w)
    }


def loss_physics(reconstructed, blurred_gt, Cn2_field, forward_model, params):
    B, C, H, W = reconstructed.shape
    if C == 3:
        reconstructed = torch.mean(reconstructed, dim=1, keepdim=True)
        blurred_gt = torch.mean(blurred_gt, dim=1, keepdim=True)
    
    re_degraded_list = []
    
    for b in range(B):
        cn2_single = Cn2_field[b]
        
        turb_params = prepare_turbulence_params_from_cn2(
            cn2_single,
            img_shape=(H, W),
            params=params,
            device=reconstructed.device
        )
        
        re_degraded = forward_model(
            reconstructed[b:b+1],  # (1, 1, H, W)
            turb_params['gray_coeff'],  # (1, 1, H, W)
            turb_params['disp_x'],  # (1, H, W)
            turb_params['disp_y'],  # (1, H, W)
            turb_params['psf']  # (1, n_h, n_w, psf_h, psf_w)
        )
        re_degraded_list.append(re_degraded)
    
    re_degraded_batch = torch.cat(re_degraded_list, dim=0)  # (B, 1, H, W)
    loss = torch.mean(torch.abs(re_degraded_batch - blurred_gt))
    
    return loss