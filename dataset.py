import random
from os.path import join
import os
import cv2
import numpy as np
import scipy.io as scio
import torch
from torch.utils.data import Dataset

def get_files(path):
    files = os.listdir(path)
    files.sort(key=lambda x: int(x[:-4]))
    return files

def shuffle_list(num):
    lis = list(range(num))
    random.shuffle(lis)
    return lis

class TrainDataset(Dataset):
    def __init__(self, para, data_type = 'train/', crop_size=None):
        self.clean_videos_path = join(para.data_root, data_type, 'clean_sequences/')
        self.turbulent_videos_path = join(para.data_root, data_type, 'turbulence_sequences/')
        self.para_labels_path = join(para.data_root, data_type, 'TS_label/')
        self.turbulent_videos = get_files(self.turbulent_videos_path)
        self.videos_num = len(self.turbulent_videos)
        self.path_list = shuffle_list(self.videos_num)

        self.H, self.W = 256, 256
        self.N_frames = para.frame_length
        self.block_size = 16
        self.para_h = self.H // self.block_size
        self.para_w = self.W // self.block_size

        self.crop_size = crop_size 
        if crop_size is not None:
            assert crop_size % self.block_size == 0, "crop_size must be N * block_size"
            self.crop_para_size = crop_size // self.block_size

    def __getitem__(self, idx):
        seq_idx = self.path_list[idx]
        ts_idx = self.path_list[idx]
        input_data, gt_data, para_data = self.get_data(seq_idx, ts_idx)
        return input_data, gt_data, para_data

    def __len__(self):
        return self.videos_num

    def get_data(self, seq_idx, ts_idx):
        gt_video_path = self.clean_videos_path + '{:04d}.avi'.format(seq_idx)
        input_video_path = self.turbulent_videos_path + '{:04d}.avi'.format(seq_idx)
        para_path = self.para_labels_path + '{:04d}.mat'.format(ts_idx)

        gt_video = cv2.VideoCapture(gt_video_path)
        input_video = cv2.VideoCapture(input_video_path)

        gt_data = np.zeros((self.N_frames, self.H, self.W, 1), dtype=np.uint8)
        input_data = np.zeros((self.N_frames, self.H, self.W, 1), dtype=np.uint8)
        para_data = np.zeros((5, self.para_h, self.para_w, self.N_frames))

        for i in range(self.N_frames):
            rval1, gt_frame = gt_video.read()
            rval2, input_frame = input_video.read()
            gt_gray = cv2.cvtColor(gt_frame, cv2.COLOR_BGR2GRAY)
            input_gray = cv2.cvtColor(input_frame, cv2.COLOR_BGR2GRAY)
            gt_data[i, :, :, 0] = gt_gray
            input_data[i, :, :, 0] = input_gray

        label_dic = scio.loadmat(para_path)
        para_dic = label_dic['Turbu_Mat']

        para_data[0, :, :, :] = para_dic[:, :, 0, :] / (8.32e-12)
        para_data[1, :, :, :] = para_dic[:, :, 2, :] / 14
        para_data[2, :, :, :] = para_dic[:, :, 1, :] / 320
        para_data[3, :, :, :] = para_dic[:, :, 3, :] / 8200
        para_data[4, :, :, :] = para_dic[:, :, 4, :] / 88

        if self.crop_size is not None:
            max_y = self.H - self.crop_size
            max_x = self.W - self.crop_size
            y0 = random.randint(0, max_y)
            x0 = random.randint(0, max_x)
            y1 = y0 + self.crop_size
            x1 = x0 + self.crop_size

            gt_data = gt_data[:, y0:y1, x0:x1, :]
            input_data = input_data[:, y0:y1, x0:x1, :]

            py0 = y0 // self.block_size
            px0 = x0 // self.block_size
            py1 = py0 + self.crop_para_size
            px1 = px0 + self.crop_para_size
            para_data = para_data[:, py0:py1, px0:px1, :]

        gt_data = torch.from_numpy(gt_data).float() / 255.0
        input_data = torch.from_numpy(input_data).float() / 255.0
        gt_data = gt_data.permute(0, 3, 1, 2)
        input_data = input_data.permute(0, 3, 1, 2)
        para_data = torch.from_numpy(para_data).float().permute(3, 0, 1, 2)

        return input_data, gt_data, para_data



class ValidDataset(Dataset):
    def __init__(self, para, data_type = 'test/', crop_size=None):
        self.clean_videos_path = join(para.data_root, data_type, 'clean_sequences/')
        self.turbulent_videos_path = join(para.data_root, data_type, 'turbulence_sequences/')
        self.para_labels_path = join(para.data_root, data_type, 'TS_label/')
        self.turbulent_videos = get_files(self.turbulent_videos_path)
        self.videos_num = len(self.turbulent_videos)
        self.path_list = list(range(self.videos_num))

        self.H, self.W = 256, 256
        self.N_frames = para.frame_length
        self.block_size = 16
        self.para_h = self.H // self.block_size
        self.para_w = self.W // self.block_size

        self.crop_size = crop_size 
        if crop_size is not None:
            assert crop_size % self.block_size == 0, "crop_size must be N * block_size"
            self.crop_para_size = crop_size // self.block_size

    def __getitem__(self, idx):
        seq_idx = self.path_list[idx]
        ts_idx = self.path_list[idx]
        input_data, gt_data, para_data = self.get_data(seq_idx, ts_idx)
        return input_data, gt_data, para_data

    def __len__(self):
        return self.videos_num

    def get_data(self, seq_idx, ts_idx):
        gt_video_path = self.clean_videos_path + '{:04d}.avi'.format(seq_idx)
        input_video_path = self.turbulent_videos_path + '{:04d}.avi'.format(seq_idx)
        para_path = self.para_labels_path + '{:04d}.mat'.format(ts_idx)

        gt_video = cv2.VideoCapture(gt_video_path)
        input_video = cv2.VideoCapture(input_video_path)

        gt_data = np.zeros((self.N_frames, self.H, self.W, 1), dtype=np.uint8)
        input_data = np.zeros((self.N_frames, self.H, self.W, 1), dtype=np.uint8)
        para_data = np.zeros((5, self.para_h, self.para_w, self.N_frames))

        for i in range(self.N_frames):
            rval1, gt_frame = gt_video.read()
            rval2, input_frame = input_video.read()
            gt_gray = cv2.cvtColor(gt_frame, cv2.COLOR_BGR2GRAY)
            input_gray = cv2.cvtColor(input_frame, cv2.COLOR_BGR2GRAY)
            gt_data[i, :, :, 0] = gt_gray
            input_data[i, :, :, 0] = input_gray

        label_dic = scio.loadmat(para_path)
        para_dic = label_dic['Turbu_Mat']

        para_data[0, :, :, :] = para_dic[:, :, 0, :] / (8.32e-12)
        para_data[1, :, :, :] = para_dic[:, :, 2, :] / 14
        para_data[2, :, :, :] = para_dic[:, :, 1, :] / 320
        para_data[3, :, :, :] = para_dic[:, :, 3, :] / 8200
        para_data[4, :, :, :] = para_dic[:, :, 4, :] / 88

        if self.crop_size is not None:
            max_y = self.H - self.crop_size
            max_x = self.W - self.crop_size
            y0 = random.randint(0, max_y)
            x0 = random.randint(0, max_x)
            y1 = y0 + self.crop_size
            x1 = x0 + self.crop_size
            gt_data = gt_data[:, y0:y1, x0:x1, :]
            input_data = input_data[:, y0:y1, x0:x1, :]

            py0 = y0 // self.block_size
            px0 = x0 // self.block_size
            py1 = py0 + self.crop_para_size
            px1 = px0 + self.crop_para_size
            para_data = para_data[:, py0:py1, px0:px1, :]

        gt_data = torch.from_numpy(gt_data).float() / 255.0
        input_data = torch.from_numpy(input_data).float() / 255.0
        gt_data = gt_data.permute(0, 3, 1, 2)
        input_data = input_data.permute(0, 3, 1, 2)
        para_data = torch.from_numpy(para_data).float().permute(3, 0, 1, 2)

        return input_data, gt_data, para_data