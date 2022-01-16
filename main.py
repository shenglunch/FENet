from __future__ import print_function, division
import argparse
import os

from matplotlib.pyplot import axes, axis
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
from models.loss import crossentropy_loss, smooth_l1_loss, model_loss
from utils import *
from torch.utils.data import DataLoader
import gc

import torchvision
from os.path import join, split, isdir, isfile, splitext, split, abspath, dirname
import skimage
from skimage import io

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='CAL-Net')
parser.add_argument('--model', help='select a model structure')
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')

parser.add_argument('--dataset', required=True, help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', required=True, help='data path')
parser.add_argument('--trainlist', required=True, help='training list')
parser.add_argument('--testlist', required=True, help='testing list')

parser.add_argument('--batch_size', type=int, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=8, help='testing batch size')
parser.add_argument('--lr', type=float, default=0.001, help='base learning rate')
parser.add_argument('--epochs', type=int, required=True, help='number of epochs to train')
parser.add_argument('--warmupepochs', type=str, required=True, help='warmupepochs')
parser.add_argument('--lrepochs', type=str, required=True, help='the epochs to decay lr: the downscale rate')

parser.add_argument('--logdir', required=True, help='the directory to save logs and checkpoints')
parser.add_argument('--load_ckpt', help='load the weights from a specific checkpoint')
parser.add_argument('--resume', action='store_true', help='continue training the model')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

parser.add_argument('--sumr_freq', type=int, default=20, help='the frequency of saving summary')
parser.add_argument('--save_freq', type=int, default=1, help='the frequency of saving checkpoint')
parser.add_argument('--save_path', type=str, default='error/',
                    help='the path of saving checkpoints and log')
# parse arguments, set seeds
args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
os.makedirs(args.logdir, exist_ok=True)

# create summary logger
print("creating new summary file")
logger = SummaryWriter(args.logdir)

# dataset, dataloader
stereo_dataset = __datasets__[args.dataset]
train_dataset = stereo_dataset(args.datapath, args.trainlist, True)
test_dataset = stereo_dataset(args.datapath, args.testlist, False)
train_img_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=8, drop_last=True)
test_img_loader = DataLoader(test_dataset, args.test_batch_size, shuffle=False, num_workers=4, drop_last=False)

# model, optimizer
model = CalNet(args.maxdisp)
model = nn.DataParallel(model)
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

parameters = filter(lambda p: p.requires_grad, model.parameters())
parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
print('\t==> trainable parameters: %.4fM' % parameters)

# load parameters
start_epoch = 0
if args.resume:
    # find all checkpoints file and sort according to epoch id
    all_saved_ckpts = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
    all_saved_ckpts = sorted(all_saved_ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # use the latest checkpoint file
    load_ckpt = os.path.join(args.logdir, all_saved_ckpts[-1])
    print("loading the lastest model in logdir: {}".format(load_ckpt))
    state_dict = torch.load(load_ckpt)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    start_epoch = state_dict['epoch'] + 1
elif args.load_ckpt:
    # load the checkpoint file specified by args.load_ckpt
    print("loading model {}".format(args.load_ckpt))
    state_dict = torch.load(args.load_ckpt)
    model.load_state_dict(state_dict['model'])
    start_epoch = state_dict['epoch'] + 1
    start_epoch = 0
print("start at epoch {}".format(start_epoch))

def train():
    for epoch_idx in range(start_epoch, args.epochs):
        print("epoch_idx", epoch_idx)
        adjust_learning_rate(optimizer, epoch_idx, args.lr, args.warmupepochs, args.lrepochs)
        is_bn = True
        if epoch_idx >= 750:
            is_bn=False

        # training
        train_img_loader=[]
        for batch_idx, sample in enumerate(train_img_loader):
            global_step = len(train_img_loader) * epoch_idx + batch_idx
            start_time = time.time()
            do_summary = global_step % args.sumr_freq == 0
            loss, scalar_outputs, image_outputs = train_sample(sample, is_bn, compute_metrics=do_summary)
            if do_summary:
                save_scalars(logger, 'train', scalar_outputs, global_step)
                save_images(logger, 'train', image_outputs, global_step)
            del scalar_outputs, image_outputs
            print('Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f}'.format(epoch_idx, args.epochs,
                                                                                       batch_idx,
                                                                                       len(train_img_loader), loss,
                                                                                       time.time() - start_time))
        # saving checkpoints
        if (epoch_idx + 1) % args.save_freq == 0:
            ckpt_data = {'epoch': epoch_idx, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(ckpt_data, "{}/checkpoint_{:0>6}.ckpt".format(args.logdir, epoch_idx))
        gc.collect()

        # testing
        avg_test_scalars = AverageMeterDict()
        #xx = [110, 4119]#[0, 27, 36, 54, 110, 193, 232, 270, 279, 323, 395, 508, 537, 1300, 1303, 1335, 1351, 1402, \
            #1608, 1609, 3120, 3117, 4059, 4066, 4119, 4191, 4209]
        #[710, 743,765,4290,4296]
        for batch_idx, sample in enumerate(test_img_loader):
            global_step = len(test_img_loader) * epoch_idx + batch_idx
            start_time = time.time()
            do_summary = global_step % args.sumr_freq == 0
            # if batch_idx in xx:
            loss, scalar_outputs, image_outputs = test_depth_sample(sample, batch_idx, compute_metrics=do_summary)
            if do_summary:
                save_scalars(logger, 'test', scalar_outputs, global_step)
                save_images(logger, 'test', image_outputs, global_step)
            avg_test_scalars.update(scalar_outputs)
            del scalar_outputs, image_outputs
            print('Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(epoch_idx, args.epochs,
                                                                                    batch_idx,
                                                                                    len(test_img_loader), loss,
                                                                                    time.time() - start_time))
        avg_test_scalars = avg_test_scalars.mean()
        save_scalars(logger, 'fulltest', avg_test_scalars, epoch_idx)
        print("avg_test_scalars", avg_test_scalars)
        gc.collect()

def train_sample(sample, is_bn, compute_metrics=False):
    model.train()
    # for m1 in model.modules():
    #     if isinstance(m1, torch.nn.BatchNorm2d):
    #         print('bn2d', m1.running_mean, m1.running_var)
    #         break 
    if not is_bn:
        model.apply(fix_bn)

    LImg, RImg, disp_gt = sample['left'], sample['right'], sample['disparity']
    LImg = LImg.cuda()
    RImg = RImg.cuda()
    disp_gt = disp_gt.cuda()
    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)

    optimizer.zero_grad()

    crossentropy, smooth_l1, disp_ests = model(LImg, RImg, disp_gt, mask) # gamma
    
    weights1 = [1.0, 1.0, 1.0, 1.0, 1.0]#[0.5, 0.5, 0.75, 1.0, 1.0]
    weights2 = [1.0, 1.0, 1.0, 1.0, 1.0]
    loss1 = []
    loss2 = []
    for i, (l1, w1) in enumerate(zip(crossentropy, weights1)):
        loss1.append(w1*l1)
    for i, (l2, w2) in enumerate(zip(smooth_l1, weights2)):
        loss2.append(w2*l2)
    loss = sum(sum(loss1) + sum(loss2))

    scalar_outputs = {"loss": loss}
    image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "LImg": LImg, "RImg": RImg}
    if compute_metrics:
        with torch.no_grad():
            image_outputs["errormap"] = [disp_error_image_func(disp_est, disp_gt) for disp_est in disp_ests]
            scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    loss.backward()
    optimizer.step()

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs

@make_nograd_func
def test_sample(sample, batch_idx, compute_metrics=True):
    model.eval()

    LImg, RImg, disp_gt = sample['left'], sample['right'], sample['disparity']
    LImg = LImg.cuda()
    RImg = RImg.cuda()
    disp_gt = disp_gt.cuda()

    start_time = time.time()
    disp_ests, residuals, similarity3 = model(LImg, RImg, disp_gt)
    torch.cuda.synchronize()
    time_ome = time.time() - start_time

    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
    loss,_ = model_loss(disp_ests, disp_gt, mask)

    scalar_outputs = {"loss": loss}#
    image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "LImg": LImg, "RImg": RImg}
    image_outputs["errormap"] = [disp_error_image_func(disp_est, disp_gt) for disp_est in disp_ests]
    scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    # scalar_outputs["Thres005"] = [Thres_metric(disp_est, disp_gt, mask, 0.05) for disp_est in disp_ests]
    # scalar_outputs["Thres01"] = [Thres_metric(disp_est, disp_gt, mask, 0.1) for disp_est in disp_ests]
    # scalar_outputs["Thres05"] = [Thres_metric(disp_est, disp_gt, mask, 0.5) for disp_est in disp_ests]
    scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
    scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
    scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]
    scalar_outputs["Thres5"] = [Thres_metric(disp_est, disp_gt, mask, 5.0) for disp_est in disp_ests]
    # scalar_outputs["Thres10"] = [Thres_metric(disp_est, disp_gt, mask, 10.0) for disp_est in disp_ests]
    # scalar_outputs["Time"] = [time_ome]
    
    # print_err(scalar_outputs, batch_idx)
    # save_disp(disp_ests[0].detach().cpu(), batch_idx, 'CD.png')
    # save_disp(disp_ests[1].detach().cpu(), batch_idx, 'D.png')
    # save_disp(disp_gt.detach().cpu(), batch_idx, 'G.png')
    # save_a1(a1.detach().cpu(), batch_idx)
    # save_a2(a2.detach().cpu(), batch_idx)
    # save_err(image_outputs["errormap"][-1].detach().cpu(), batch_idx)
    # save_sim(similarity3.detach().cpu(), batch_idx, 'S.png')
    # save_f(residuals[-1].squeeze(0).cpu().numpy(), batch_idx, 'R.png')
    # save_f(eb.squeeze(0).cpu().numpy(), batch_idx, 'EB.png')
    # save_f(e_f.squeeze(0).detach().cpu().numpy(), batch_idx, 'EF.png')
    # save_f(e_rgb.squeeze(0).detach().cpu().numpy(), batch_idx, 'ER.png')
    # save_f(e_disp.squeeze(0).detach().cpu().numpy(), batch_idx, 'ED.png')
    # save_f(fd2.squeeze(0).detach().cpu().numpy(), batch_idx, 'F.png')

    # depth_ests = []
    # for disp in disp_ests:
    #     mask = disp <= 0
    #     depth_ests.append(1050*0.27 / disp)
    #     depth_ests[-1][mask] = 0.0

    # mask = disp_gt <= 0
    # depth_gt = 1050*0.27 / disp_gt
    # depth_gt[mask] = 0.0
    # for i, depth in enumerate(depth_ests):
    #     GD, ARD = GD_metric(depth, depth_gt)
    #     scalar_outputs["GD_"+str(i)] = [GD]
    #     scalar_outputs["ARD_"+str(i)] = ARD
    #     mae, rmse, mae_p = Mae_metric(depth, depth_gt)
    #     scalar_outputs["MAE_"+str(i)] = [mae]
    #     scalar_outputs["RMSE_"+str(i)] = [rmse]
    #     scalar_outputs["MAE_P_"+str(i)] = mae_p
    #     mask = (depth_gt <= 84) & (depth_gt >=4)
    #     scalar_outputs["Out005_"+str(i)] = Outlier_metric(depth, depth_gt, mask=mask)
        
    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs

@make_nograd_func
def test_depth_sample(sample, batch_idx, compute_metrics=True):
    model.eval()

    LImg, RImg, disp_gt, depth_gt, bf = sample['left'], sample['right'], sample['disparity'], sample['depth'], sample['bf']
    LImg = LImg.cuda()
    RImg = RImg.cuda()
    disp_gt = disp_gt.cuda()
    depth_gt = depth_gt.cuda()
    bf = bf.cuda()

    start_time = time.time()
    disp_ests, residuals, similarity3 \
         = model(LImg, RImg, disp_gt)
    torch.cuda.synchronize()
    time_ome = time.time() - start_time

    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
    loss,_ = model_loss(disp_ests, disp_gt, mask)

    scalar_outputs = {"loss": loss}
    image_outputs = {"RImg": RImg}
    image_outputs["errormap"] = [disp_error_image_func(disp_est, disp_gt) for disp_est in disp_ests]
    scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["Thres005"] = [Thres_metric(disp_est, disp_gt, mask, 0.05) for disp_est in disp_ests]
    scalar_outputs["Thres01"] = [Thres_metric(disp_est, disp_gt, mask, 0.1) for disp_est in disp_ests]
    scalar_outputs["Thres05"] = [Thres_metric(disp_est, disp_gt, mask, 0.5) for disp_est in disp_ests]
    scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
    scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
    scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]
    scalar_outputs["Thres5"] = [Thres_metric(disp_est, disp_gt, mask, 5.0) for disp_est in disp_ests]
    scalar_outputs["Thres10"] = [Thres_metric(disp_est, disp_gt, mask, 10.0) for disp_est in disp_ests]
    scalar_outputs["Time"] = [time_ome]

    print_err(scalar_outputs, batch_idx)
    save_err(image_outputs["errormap"][-1].detach().cpu(), batch_idx, "Ei.jpg")
    save_disp(disp_ests[1].detach().cpu(), batch_idx, 'Di.png')

    depth_ests = []
    for disp in disp_ests:
        mask = disp <= 0
        depth_ests.append(bf / disp)
        depth_ests[-1][mask] = 0.0
    
    save_disp(depth_ests[1].detach().cpu(), batch_idx, 'De.png')
        
    mask = disp_gt <= 0
    depth_gt = bf / disp_gt
    depth_gt[mask] = 0.0
    for i, depth in enumerate(depth_ests):
        GD, ARD = GD_metric(depth, depth_gt)
        scalar_outputs["GD_"+str(i)] = [GD]
        scalar_outputs["ARD_"+str(i)] = ARD
        save_err(disp_error_image_func(depth, depth_gt).detach().cpu(), batch_idx, "Ee.jpg")
    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs

def save_disp(disp, batch_idx, f):
    name = join(args.save_path+"ds/D/%04d-" % batch_idx + f)
    disp = np.array(disp.squeeze(), dtype=np.float32)
    disp = np.round(disp * 256).astype(np.uint16)
    skimage.io.imsave(name, disp)

def save_gt(disp, batch_idx):
    name = join(args.save_path+"kitti/%04d-" % batch_idx + "G.png")
    disp = np.array(disp.squeeze(), dtype=np.float32)
    disp = np.round(disp * 256).astype(np.uint16)
    skimage.io.imsave(name, disp)

def save_err(err, batch_idx, f):
    torchvision.utils.save_image(err, \
        join(args.save_path+"ds/E/%04d-" % batch_idx + f))

def print_err(errs, batch_idx):
    print('No.%04d'%batch_idx + ', EPE:{:.5f}, D1:{:.5f}'.format(errs['EPE'][-1], errs['D1'][-1]))

def save_a1(a1, batch_idx): #channel
    np.save(join(args.save_path+"kitti/%04d-" % batch_idx + "AC.npy"), a1)

def save_a2(a2, batch_idx): #spatial
    np.save(join(args.save_path+"kitti/%04d-" % batch_idx + "AS.npy"), a2)

def save_r(x, batch_idx):
    ma = float(x.max())
    mi = float(x.min())
    d = ma-mi if ma!=mi else 1e5
    x = (x-mi)/d
    name = join(args.save_path+"test/%04d-" % batch_idx + "R.png")
    x = np.array(x.squeeze(), dtype=np.float32)
    x = np.round(x).astype(np.uint16)
    skimage.io.imsave(name, x)

def save_sim(x, batch_idx, f):
    import matplotlib.pyplot as plt
    import matplotlib 
    matplotlib.use('pdf')

    for i in range(x.shape[1]):
        x1 = x[0,i,:,:]
        # x = np.mean(x, axis=0)
        plt.imshow(x1) 
        plt.axis('off')
        plt.savefig(join(args.save_path+"sceneflow/%04d-" % batch_idx + str(i) +f), bbox_inches = 'tight', pad_inches = 0, dpi=193.5485)

def save_f(x, batch_idx, f):
    import matplotlib.pyplot as plt
    import matplotlib 
    matplotlib.use('pdf')
    
    x = np.mean(x, axis=0)

    plt.imshow(x) 
    plt.axis('off')
    plt.savefig(join(args.save_path+"kitti/%04d-" % batch_idx + f), bbox_inches = 'tight', pad_inches = 0, dpi=193.5485)
    # np.save(join(args.save_path+"sceneflow/%04d-" % batch_idx + f), x)

def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

if __name__ == '__main__':
    train()
