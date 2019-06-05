from __future__ import print_function, absolute_import
import torch
import os
import scipy.io as sio
from torch.utils.data import Dataset


class ToTensor(object):
    def __call__(self, sample):
        y, k, kt, x_gt = sample['y'], sample['k'], sample['kt'], sample['x_gt']
        img_ch_num = len(y.shape)
        if img_ch_num == 2:
            x0 = y
            x_gt = x_gt
            y = y
        elif img_ch_num == 3:
            x0 = y
            x0 = x0.transpose(2, 0, 1)
            x_gt = x_gt.transpose((2, 0, 1))
            y = y.transpose((2, 0, 1))

        return torch.from_numpy(y).float(), torch.from_numpy(x_gt).float(),\
            torch.from_numpy(k.reshape(1, k.shape[0], k.shape[1])).float(), \
            torch.from_numpy(kt.reshape(1, k.shape[0], k.shape[1])).float(), \
            torch.from_numpy(x0).float()


class BlurryImageDataset(Dataset):
    """Blur image dataset"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_name_list = [
            name for name in os.listdir(self.root_dir)
            if os.path.isfile(os.path.join(self.root_dir, name)) and name.endswith('.mat')
        ]
        print(self.file_name_list)
        self.file_name_list.sort()
        self.TensorConverter = ToTensor()

    def __len__(self):
        return len([name for name in os.listdir(self.root_dir) \
                    if os.path.isfile(os.path.join(self.root_dir, name)) and name.endswith('.mat') ])

    def __getitem__(self, idx):
        """get .mat file"""
        mat_name = self.file_name_list[idx]
        sample = sio.loadmat(os.path.join(self.root_dir, mat_name))
        if self.transform:
            sample = self.transform(sample)

        return self.TensorConverter(sample), mat_name