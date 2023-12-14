from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

def subpixel(disp, costFive0, costFive1, costFive2, costFive3, costFive4, maxdisp=192):
    with torch.no_grad():
        tmp1 = (costFive1 - costFive3)
        tmp2 = (costFive1 - costFive2 - costFive2 + costFive3)
        tmp3 = tmp1 - (costFive2 - costFive4)
        tmp4 = (costFive0 - costFive2) - tmp1
        d_sub = torch.zeros_like(disp, dtype=disp.dtype, device=disp.device)
        mask = ((tmp1 > 0) & (tmp2 > 0)) | ((tmp1 < 0)  & (tmp2 < 0))
        d_temp = ((tmp1 / (tmp2 + tmp2+0.0000001)) + (tmp1 / (tmp3+0.00000001))) / 2
        d_sub[mask]=d_temp[mask]
        d_temp = (( tmp1 / (tmp2 + tmp2+0.00000001)) + ( tmp1 / (tmp4+0.00000001))) / 2
        mask = ~mask
        d_sub[mask] = d_temp[mask]

        mask = (disp==0)|(disp==maxdisp-1)
        d_sub[mask]=0

    return d_sub+disp

def predict_disparity(cost, maxdisp=192): 
    '''
      input
        cost: b, d, h, w
      output
        pre_disp: b, h, w
    '''
    with torch.no_grad():
        # integer disparity
        disp = torch.unsqueeze(cost, 1).argmax(2)
        # window disparity
        disp_1 = disp - 1
        disp_1 = torch.where(disp_1 < 0, torch.full_like(disp_1, 0), disp_1)
        disp_2 = disp - 2
        disp_2 = torch.where(disp_2 < 0, torch.full_like(disp_2, 0), disp_2)
        disp1 = disp + 1
        disp1 = torch.where(disp1 > (maxdisp - 1), torch.full_like(disp1, maxdisp - 1), disp1)
        disp2 = disp + 2
        disp2 = torch.where(disp2 > (maxdisp - 1), torch.full_like(disp2, maxdisp - 1), disp2)
        # disparity probability
        pro_1 = torch.gather(cost, 1, disp_1)
        pro_2 = torch.gather(cost, 1, disp_2)
        pro = torch.gather(cost, 1, disp)
        pro1 = torch.gather(cost, 1, disp1)
        pro2 = torch.gather(cost, 1, disp2)
        # subpixel
        pre_disp = subpixel(disp.float(), 1 - pro_2, 1 - pro_1, 1 - pro, 1 - pro1, 1 - pro2, maxdisp)

    return pre_disp

def crossentropy_loss(costs, preds, gt, mask):
    cost_gt = torch.unsqueeze(gt, 1)
    bins = torch.zeros_like(costs[0]).cuda()
    for i in range(costs[0].shape[1]):
        bins[:, i, :, :] = i
    res = (bins - cost_gt).float()
    cost_gt = torch.exp(-torch.pow(res, 2)).float()
    cost_gt = F.softmax(-torch.abs(res), dim=1)

    crossentropy = []
    for i, (cost, pred) in enumerate(zip(costs, preds)):
        with torch.no_grad():
            diff = 1 + torch.exp(-(pred - gt).abs())
        cost = torch.log(cost + 1e-6)
        cost = cost.mul(cost_gt)
        cost = cost.sum(1) * diff
        crossentropy.append(-cost[mask].mean())
    return crossentropy

def smooth_l1_loss(refine, confs, coarse, disp_gt, mask):
    losses = []
    with torch.no_grad():
        diff = 1 + torch.exp(-(coarse[mask] - disp_gt[mask]).abs())
        
    for i, (disp_est, conf) in enumerate(zip(refine, confs)):
        conf = conf[mask]
        # conf_exp = torch.exp(conf)
        l1 = F.smooth_l1_loss(disp_est[mask]*diff, disp_gt[mask]*diff, reduction='mean')
        reg = torch.log(conf + 1e-6).mean()
        losses.append(l1+reg*1e-1)
    return losses

class Conv2D(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, groups=1, bias=False, activation='relu', norm='batch'):
        super(Conv2D, self).__init__()
        self.conv = nn.Conv2d(input_size, output_size, kernel_size, stride, padding, groups=groups, bias=bias)

        self.norm = norm
        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'relu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out

class Deconv2D(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, bias=False, activation='relu', norm='batch'):
        super(Deconv2D, self).__init__()
        self.deconv = nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'relu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out

class DilaConv2D(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, dilation=2, bias=False, activation='relu', norm='batch'):
        super(DilaConv2D, self).__init__()
        self.conv = nn.Conv2d(input_size, output_size, kernel_size, stride, padding, dilation, bias=bias)

        self.norm = norm
        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'relu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out

class Conv3D(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=False, activation='relu', norm='batch'):
        super(Conv3D, self).__init__()
        self.conv = nn.Conv3d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm3d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm3d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'relu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out

class Deconv3D(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, output_padding=1, bias=False, activation='relu', norm='batch'):
        super(Deconv3D, self).__init__()
        self.deconv = nn.ConvTranspose3d(input_size, output_size, kernel_size, stride, padding, output_padding=output_padding, bias=False)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm3d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm3d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'relu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation, activation='relu', norm='batch'):
        super(BasicBlock, self).__init__()

        self.conv1 = Conv2D(inplanes, planes, 3, stride, pad, dilation, activation=activation, norm=norm)
        self.conv2 = Conv2D(planes, planes, 3, 1, pad, dilation, activation=None, norm=norm)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out

class ShuffleBlock(nn.Module):
    def __init__(self, inp, oup, stride, benchmodel, activation='relu', norm='batch'):
        super(ShuffleBlock, self).__init__()
        self.benchmodel = benchmodel
        self.stride = stride
        assert stride in [1, 2]

        oup_inc = oup//2
        
        if self.benchmodel == 1:
        	self.banch2 = nn.Sequential(
                Conv2D(oup_inc, oup_inc, 1, 1, 0, activation=activation, norm=norm), # pw
                Conv2D(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, activation=None, norm=norm), # dw
                Conv2D(oup_inc, oup_inc, 1, 1, 0, activation=activation, norm=norm), # pw-linear
            )                
        else:                  
            self.banch1 = nn.Sequential(
                Conv2D(inp, inp, 3, stride, 1, groups=inp, activation=None, norm=norm), # dw
                Conv2D(inp, oup_inc, 1, 1, 0, activation=activation, norm=norm), # pw-linea
            )       
            self.banch2 = nn.Sequential(
                Conv2D(inp, oup_inc, 1, 1, 0, activation=activation, norm=norm), # pw
                Conv2D(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, activation=None, norm=norm), # dw
                Conv2D(oup_inc, oup_inc, 1, 1, 0, activation=activation, norm=norm), # pw-linea
            )     
    
    def channel_shuffle(self, x, groups):
        b, c, h, w = x.data.size()
        channels_per_group = c // groups 
        # reshape
        x = x.view(b, groups, channels_per_group, h, w)
        x = torch.transpose(x, 1, 2).contiguous()
        # flatten
        x = x.view(b, -1, h, w)
        return x
        
    def forward(self, x):
        if self.benchmodel == 1:
            x1 = x[:, :(x.shape[1]//2), :, :]
            x2 = x[:, (x.shape[1]//2):, :, :]
            x2 = self.banch2(x2)
            out = torch.cat((x1, x2), dim=1)
        elif self.benchmodel == 2:
            x1 = self.banch1(x)
            x2 = self.banch2(x)
            out = torch.cat((x1, x2), dim=1)
        
        out = self.channel_shuffle(out, 2)
        return out

class SEBlock(nn.Module):
    def __init__(self, inplanes, scale=16):
        super(SEBlock, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=inplanes, out_features=round(inplanes / scale)),
            nn.ReLU(True))
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=round(inplanes / scale), out_features=inplanes),
            nn.Sigmoid())

    def forward(self, x): 
        b, c = x.size(0), x.size(1)
        x1 = self.gap(x).view(b, -1)
        x1 = self.fc1(x1)
        s = self.fc2(x1).view(b, c, 1, 1)
        y = x + x*s
        return y, F.relu(x, inplace=False)

class EdgeBlock(nn.Module):
    def __init__(self, planes, activation='relu', norm='batch', bias=False):
        super(EdgeBlock, self).__init__()
        
        self.convh = DilaConv2D(planes, planes*2, 3, 1, 2, 2, bias=bias, activation=activation, norm=norm)
        self.convl = DilaConv2D(planes*2, planes*2, 3, 1, 2, 2, bias=bias, activation=activation, norm=norm) 
        self.conve = Conv2D(planes*2, planes, 3, 1, 1, bias=bias, activation='sigmoid', norm=norm)
    
    def forward(self, h, l):
        h = self.convh(h)
        l = self.convl(l)
        l_h = F.interpolate(l, [h.size()[2], h.size()[3]], mode='bilinear', align_corners=True)
        e = self.conve(h - l_h)
        return e, F.relu(h - l_h, inplace=False)

class FeedbackBlock(nn.Module):
    def __init__(self, in_filter, num_filter, kernel_size=8, stride=4, padding=2, bias=False, activation='relu', norm='batch'):
        super(FeedbackBlock, self).__init__()
        self.down1 = Conv2D(in_filter, num_filter, kernel_size, stride, padding, bias=bias, activation=activation, norm=norm)
        self.conv1 = Conv2D(in_filter, num_filter, 1, 1, 0, bias=bias, activation=activation, norm=norm)
        self.avgpool_1 = nn.AvgPool2d(4, 4, 0)
        self.up_1 = Deconv2D(num_filter, num_filter, 8, 4, 2, bias=bias, activation=activation, norm=norm)
        self.act_1 = nn.ReLU(True)
        self.up_conv1 = Deconv2D(num_filter, num_filter, kernel_size, stride, padding, bias=bias, activation=activation, norm=norm)
        self.up_conv2 = Conv2D(num_filter, num_filter, kernel_size, stride, padding, bias=bias, activation=activation, norm=norm)
        self.up_conv3 = Deconv2D(num_filter, num_filter, kernel_size, stride, padding, bias=bias, activation=activation, norm=norm)

    def forward(self, l, d, e):
        return None

