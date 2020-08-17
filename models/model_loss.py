import torch
import torch.nn.functional as F
import numpy as np


class LogLoss(torch.nn.Module):
    def __init__(self):
        super(LogLoss, self).__init__()

    def forward(self, x, y):
        error = torch.pow(x - y, 2)
        sampleSize = error.size()
        error = error.view(sampleSize[0], -1)
        # print(error.size())
        mean_error = error.mean(1)
        # print(mean_error.size())
        logloss = torch.log(mean_error)
        return logloss.mean()


class ImgDiffComputer(torch.nn.Module):
    def __init__(self):
        super(ImgDiffComputer, self).__init__()

        k = np.zeros([1, 1, 2, 1], dtype=np.float32)
        k[0, 0, 0, 0] = 1
        k[0, 0, 1, 0] = -1

        self.k = torch.nn.Parameter(
            torch.from_numpy(k), requires_grad=False)

    def forward(self, x):

        xsize = x.size()

        xr = x.view(-1, 1, xsize[2], xsize[3])

        diff1 = F.conv2d(xr, self.k, padding=0)
        diff2 = F.conv2d(xr, self.k.transpose(3, 2), padding=0)

        size1 = diff1.size()
        size2 = diff2.size()

        return diff1.view(-1, 3, size1[2],
                          size1[3]), diff2.view(-1, 3, size2[2], size2[3])
