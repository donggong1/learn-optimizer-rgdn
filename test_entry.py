# script for testing a training model
# Please custumize the cropping and padding operations and stopping conditions as demanded.

from __future__ import absolute_import, print_function
import models
import torch
from torch.utils.data import DataLoader
import data
import scipy.misc
import time
import scipy.io as sio
from options.running_options import Options
import utils

#
opt_parser = Options()
opt = opt_parser.parse(is_print=True)
use_cuda = opt.UseCUDA
device = torch.device("cuda" if use_cuda else "cpu")
crop_size = opt.CropSize # set as 0 if input is not padded in advance

# model
trained_model = models.OptimizerRGDN(
        opt.StepNum,
        use_grad_adj=opt.UseGradAdj,
        use_grad_scaler=opt.UseGradScaler,
        use_reg=opt.UseReg,
        stop_epsilon=opt.StopEpsilon)

model_para = torch.load(opt.TrainedModelPath, map_location=device)
trained_model.load_state_dict(model_para)
trained_model.eval()
trained_model.to(device)
##
model_name = opt.ModelName

# data path
# data_root = '../'
# dataset_name = 'rgdn_dataset'
data_path = opt.DataPath #data_root + dataset_name
outpath = opt.OutPath #data_root + dataset_name + '_results_' + model_name + '/'
utils.mkdir(outpath)

##
Dataset = data.BlurryImageDataset(data_path)
test_data_loader = DataLoader(Dataset,
                      batch_size=1,
                      shuffle=False,
                      num_workers=1)

sample_num = test_data_loader.__len__()

with torch.no_grad():
    for batch_idx, ( (y, k, kt), sample_name) in enumerate(test_data_loader):
        print('%d / %d, %s' % (batch_idx+1, sample_num, sample_name[0]))
        y, kt, k = y.to(device), k.to(device), kt.to(device)
        if(opt.ImgPad):
            k_size = k.size()[2]
            padding_size = int((k_size / 2) * 1.5)
            y = torch.nn.functional.pad(y, [padding_size, padding_size, padding_size, padding_size], mode='replicate')

        start = time.time()
        output_seq = trained_model(y, k, kt)
        # output_len = len(output_seq)
        x_final = output_seq[-1]
        # print('Time {}'.format(time.time() - start))

        ##
        if (opt.ImgPad):
            y = utils.truncate_image(y, padding_size)
            x_final = utils.truncate_image(x_final, padding_size)

        if (crop_size>0):
            x_est_np = utils.truncate_image(x_final, crop_size)
        elif(crop_size==0):
            x_est_np = x_final.cpu()
        else:
            crt_crop_size = int(k.size()[2] /2)
            x_est_np = utils.truncate_image(x_final, crt_crop_size)

        x_est_np = utils.tensor_to_np_img(x_est_np)
        #

        x_est_np = utils.box_proj(x_est_np)

        sample_name_full = sample_name[0]
        sample_name = sample_name_full[0:len(sample_name_full) - 4]

        sio.savemat(outpath + sample_name + '_estx.mat', {'x_est': x_est_np})
        scipy.misc.imsave(outpath + sample_name + '_estx.png', x_est_np * 255)
        torch.cuda.empty_cache()

