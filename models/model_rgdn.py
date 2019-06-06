from __future__ import absolute_import, print_function
import torch
from torch.nn import Module
import torch.nn.functional as F
from torch.autograd import Variable
from models.module_basic import ConvBlock

import utils

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# compute the gradient of data fitting term
class GradDataFitting(Module):
    def __init__(self):
        super(GradDataFitting, self).__init__()

    def forward(self, x, y, k, kt):
        n_size = x.size()[0]
        k_size = k.size()[2]
        padding = int(k_size / 2)
        x1 = x.transpose(1, 0)  # x1: C x N x H x W
        y1 = y.transpose(1, 0)
        # k: N x 1 x Ksize x Ksize
        vk = Variable(k.data.clone())
        vkt = Variable(kt.data.clone())
        kx_y = F.conv2d(x1, vk, padding=padding, groups=n_size) # Ax
        kx_y.sub_(y1) # Ax-y
        ktkx_kty = F.conv2d(kx_y, vkt, padding=padding, groups=n_size) # A^T(Ax-y)
        res = ktkx_kty.transpose(1, 0)  # h3: N x C x H x W
        return res


class OptimizerRGDN(Module):
    """Gradient descent based optimizer model"""
    def __init__(self, num_steps,
                 use_grad_adj=True,
                 use_grad_scaler=True,
                 use_reg=True,
                 share_parameter=True,
                 stop_epsilon = float("inf"),
                 use_cuda=True):
        super(OptimizerRGDN, self).__init__()
        #
        self.num_steps = num_steps
        self.momen = 0.8
        self.stop_epsilon = stop_epsilon
        #
        self.use_grad_adj = use_grad_adj
        self.use_reg = use_reg
        self.use_grad_scaler = use_grad_scaler
        self.share_parameter = share_parameter


        #
        self.grad_datafitting_cal = GradDataFitting()
        if self.share_parameter:
            if(self.use_reg):
                self.CNNOptimizer = ConvBlock()
            if(self.use_grad_adj):
                self.GradAdj = ConvBlock()
            if(self.use_grad_scaler):
                self.GradScaler = ConvBlock()

            # if(self.use_reg):
            #     self.CNNOptimizer.apply(weights_init)
            # if(self.all_grad_adj):
            #     self.GradScaler.apply(weights_init)
            # if(self.use_grad_adj):
            #     self.GradAdj.apply(weights_init)


    def forward(self, y, k, kt):
        # init x
        xcurrent = y
        # xcurrent = self.init_cal(xcurrent, y, k, kt)

        #
        output_list = []
        # optimization init
        for i in range(self.num_steps):
            # print(i)
            ## single step operation
            grad_loss = self.grad_datafitting_cal(xcurrent, y, k, kt)

            # H()
            if(self.use_grad_adj):
                if(self.share_parameter):
                    grad_adj = self.GradAdj(grad_loss)
            else:
                grad_adj = grad_loss

            # R(x)
            if(self.use_reg):
                if self.share_parameter:
                    grad_reg = self.CNNOptimizer(xcurrent)
                grad_direc = grad_adj - grad_reg
            else:
                grad_direc = grad_adj

            # D()
            if(self.use_grad_scaler):
                if (self.share_parameter):
                    grad_scaled = self.GradScaler(grad_direc)
            else:
                grad_scaled = grad_direc

            ## update x
            xcurrent = self.momen * xcurrent + (1 - self.momen) * grad_scaled
            ## -end- single step operation

            # output
            output_list += [xcurrent]

            ## check stopping condition, only for testing
            if(self.stop_epsilon<float("inf")):
                error = fitting_error_cal(y, xcurrent, k)
                if(i==0):
                    error_prev = error
                    error_0 = error
                else:
                    error_reltv = (abs(error-error_prev) + 1e-10) / (error_0 + 1e-10)
                    error_prev = error
                    print('ite: %d; error_reltv: %f' % (i, error_reltv))
                    if(error_reltv < self.stop_epsilon):
                        print('stop opt.')
                        break

        return output_list

def fitting_error_cal(y, x, k):
    # only used during testing
    n_size = x.size()[0]
    k_size = k.size()[2]
    padding = int(k_size / 2)
    x1 = x.transpose(1, 0)  # x1: C x N x H x W
    y1 = y.transpose(1, 0)
    # k: N x 1 x Ksize x Ksize
    vk = Variable(k.data.clone())
    kx_y = F.conv2d(x1, vk, padding=padding, groups=n_size) # Ax
    kx_y.sub_(y1) # Ax-y
    kx_y = utils.truncate_image(kx_y, k_size)

    fitting_error = torch.norm(kx_y, 'fro') / 2
    return fitting_error