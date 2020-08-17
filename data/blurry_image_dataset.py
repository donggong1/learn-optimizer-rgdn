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
            # x0 = y
            # x_gt = x_gt
            y = y
        elif img_ch_num == 3:
            # x0 = y
            # x0 = x0.transpose(2, 0, 1)
            # x_gt = x_gt.transpose((2, 0, 1))
            y = y.transpose((2, 0, 1))

        return torch.from_numpy(y).float(), \
            torch.from_numpy(k.reshape(1, k.shape[0], k.shape[1])).float(), \
            torch.from_numpy(kt.reshape(1, k.shape[0], k.shape[1])).float()


class BlurryImageDataset(Dataset):
    """Blur image dataset"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_name_list = [
            name for name in os.listdir(self.root_dir)
            if os.path.isfile(os.path.join(self.root_dir, name)) and name.endswith('.mat')
        ]
        # print(self.file_name_list)
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



# Note that this part of code (BlurryImageDatasetOnTheFly) needs to be not fully tested.
# TODO: refine the interface and details
class BlurryImageDatasetOnTheFly(Dataset):
    def __init__(self,
                 root_dir,
                 k_size=41,
                 sp_size=[11, 16, 21, 26, 31],
                 num_spl_ctrl=[3, 4, 5, 6],
                 patch_size=256,
                 max_num_images=None):
        self.root_dir = root_dir
        self.k_size = k_size
        self.sp_size = sp_size
        self.num_spl_ctrl = num_spl_ctrl
        self.patch_size = 256

        self.rksize = [
            11,
        ]

        self.file_name_list = [name for name in os.listdir(self.root_dir)
                               if os.path.isfile(os.path.join(self.root_dir, name))]
        self.file_name_list.sort()

        if max_num_images is not None and max_num_images < len(
                self.file_name_list):
            self.file_name_list = self.file_name_list[:max_num_images]

    def __len__(self):
        return len(self.file_name_list)

    def __getitem__(self, idx):
        sp_size = int(np.random.choice(self.sp_size))
        num_spl_ctrl = int(np.random.choice(self.num_spl_ctrl))
        # print(sp_size, num_spl_ctrl)
        k = kernel_sim_spline(sp_size, self.k_size, num_spl_ctrl, 1)
        k = np.reshape(k, [1, 1, self.k_size, self.k_size])

        img_name = self.file_name_list[idx]
        sample = imread(os.path.join(self.root_dir, img_name))

        if sample.shape[0] < self.patch_size or sample.shape[
                1] < self.patch_size:
            return self.__getitem__((idx - 1) % (self.__len__()))
        patches = image.extract_patches_2d(sample,
                                           [self.patch_size, self.patch_size],
                                           max_patches=1)
        sample = patches[0, ...]
        sample = sample.astype(np.float32) / 255.0
        sample = np.expand_dims(np.transpose(sample, [2, 0, 1]), 1)
        sample = torch.from_numpy(sample.astype(np.float32))  # n x c x w x h
        hks = (self.k_size) // 2

        with torch.no_grad():
            k = torch.from_numpy(k)
            y = torch.nn.functional.conv2d(sample, k)
            nl = np.random.uniform(0.003, 0.015)
            y = y + nl * torch.randn_like(y)
            y = torch.clamp(y * 255.0, 0, 255)
            y = y.type(torch.ByteTensor)
            y = y.float() / 255.0
            y = torch.nn.functional.pad(y, (hks, hks, hks, hks),
                                        mode='replicate')
            y = y.squeeze(1)
            x_gt = sample.squeeze(1)[:, hks:(-hks), hks:(-hks)]
            k = k.squeeze(0)
            kt = torch.flip(k, [1, 2])

        return y, x_gt, k, kt, y        