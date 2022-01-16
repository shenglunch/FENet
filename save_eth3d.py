from __future__ import print_function, division
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import time
from tensorboardX import SummaryWriter
from datasets import __datasets__
from models.calnet import CalNet
from utils import *
from torch.utils.data import DataLoader
import gc
import skimage
from skimage import io
import sys

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='CAL-Net')
parser.add_argument('--model', help='select a model structure')
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')

parser.add_argument('--dataset', default='eth3d', help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', required=True, help='data path')
parser.add_argument('--testlist', required=True, help='testing list')
parser.add_argument('--load_ckpt', required=True, help='load the weights from a specific checkpoint')

# parse arguments
args = parser.parse_args()

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
test_dataset = StereoDataset(args.datapath, args.testlist, False)
TestImgLoader = DataLoader(test_dataset, 1, shuffle=False, num_workers=4, drop_last=False)

# model, optimizer
model = CalNet(args.maxdisp)
model = nn.DataParallel(model)
model.cuda()

# load parameters
print("loading model {}".format(args.load_ckpt))
state_dict = torch.load(args.load_ckpt)
model.load_state_dict(state_dict['model'])

def save_pfm(file, image, scale = 1):
  color = None

  if image.dtype.name != 'float32':
    raise Exception('Image dtype must be float32.')

  if len(image.shape) == 3 and image.shape[2] == 3: # color image
    color = True
  elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
    color = False
  else:
    raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

  file.write('PF\n' if color else 'Pf\n')
  file.write('%d %d\n' % (image.shape[1], image.shape[0]))

  endian = image.dtype.byteorder

  if endian == '<' or endian == '=' and sys.byteorder == 'little':
    scale = -scale

  file.write('%f\n' % scale)

  image.tofile(file)


def test():
    #os.makedirs('./predictions', exist_ok=True)
    for batch_idx, sample in enumerate(TestImgLoader):
        start_time = time.time()
        disp_est_np = tensor2numpy(test_sample(sample))
        top_pad_np = tensor2numpy(sample["top_pad"])
        right_pad_np = tensor2numpy(sample["right_pad"])
        left_filenames = sample["left_filename"]
        ttime = time.time() - start_time
        print('Iter {}/{}, time = {:3f}'.format(batch_idx, len(TestImgLoader),
                                                ttime))

        for disp_est, top_pad, right_pad, fn in zip(disp_est_np, top_pad_np, right_pad_np, left_filenames):
            assert len(disp_est.shape) == 2
            disp_est = np.array(disp_est[top_pad:, :-right_pad], dtype=np.float32)
            #fn = os.path.join("predictions", fn.split('/')[-1])
            fn = os.path.join("PP", fn.split('/')[-2])
            print("saving to", fn, disp_est.shape)

            # invalid = np.logical_or(disp_est == np.inf, disp_est != disp_est)
            # disp_est[invalid] = np.inf

            with open('%s.pfm' % (fn), 'w') as f:
                save_pfm(f, disp_est[::-1, :])
            with open('%s.txt' % (fn), 'w') as f:
                f.write("runtime "+str(ttime))

# test one sample
@make_nograd_func
def test_sample(sample):
    model.eval()
    disp_ests, residuals = model(sample['left'].cuda(), sample['right'].cuda())
    return disp_ests[-1]


if __name__ == '__main__':
    test()
