from __future__ import absolute_import, print_function
# import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import os
# from skimage import transform

def tensor_to_np_img(tensor_img):
    tmp = (tensor_img.data).cpu().numpy()
    tmp1 = tmp[0, :, :, :]
    np_img = tmp1.transpose([1, 2, 0])
    return np_img


def truncate_image(img, s):
    # s: truncate size
    if(s>0):
        if(len(img.shape)==3):
            # F or C x H x W
            return img[:, s:(-s), s:(-s)]
        elif(len(img.shape)==4):
            # F x C x H x W
            return img[:, :, s:(-s), s:(-s)]
    else:
        return img


def tensor2numpy(tensor_in):
    """Transfer pythrch tensor to numpy array"""
    nparray_out = (Variable(tensor_in).data).cpu().numpy()
    # for different image channels
    ch_num = len(nparray_out.shape)
    # if ch_num==3:
    #     nparray_out = nparray_out.transpose((1,2,0))
    #if ch_num==4:
    #    nparray_out = nparray_out.transpose((0, 2, 3, 1))
    return nparray_out

# def show_data_sample(sample):
#     """Show data sampel: y, x_gt, k. Input is Tensor."""
#     # tmpdata = {t: (Variable(sample[t]).data).cpu().numpy()
#     tmpdata = {t: tensor2numpy(sample[t])
#                for t in ['y', 'x_gt', 'k']}
#     # for kernel
#     tmpsize = int(tmpdata['y'].shape[0])
#     tmpdata['k'] = transform.resize(tmpdata['k']/np.max(tmpdata['k']),
#                                     (tmpsize, tmpsize))
#
#     ch_num = len(tmpdata['y'].shape)
#     if ch_num==3:
#         tmpk = tmpdata['k']
#         tmpdata['k'] = np.tile(tmpk[..., None], [1, 1, 3])
#
#     print(tmpdata['x_gt'].shape)
#     plt.imshow(np.concatenate([tmpdata[t] for t in ['x_gt', 'y', 'k']], axis=1),\
#                cmap='gray')
#     # plt.show()

def transpose_kernel(k):
    """k for A(k)^T"""
    return np.fliplr(np.flipud(k))

def mkdir(path):
    """create a single empty directory if it didn't exist
    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

# project image data to [0,1]
def box_proj(input):
    output = input
    output[output>1] = 1
    output[output<0] = 0
    return output