import math
import torch
import torch.nn as nn
from .submodule import *
import numpy as np

class feature_extraction(nn.Module):
    def __init__(self, concat_feature_channel=12):
        super(feature_extraction, self).__init__()
        self.in_channel = 32

        self.conv_1 = Conv2D(3, self.in_channel, 3, 1, 1, activation='relu', norm='batch')
        # nn.Sequential(DilaConv2D(self.in_channel, self.in_channel, 3, 1, 2, 2, activation='relu', norm='batch'))
        # self.conv_2 = nn.Sequential(
        #     Conv2D(self.in_channel, self.in_channel, 3, 2, 1, activation='relu', norm='batch'),
        #     Conv2D(self.in_channel, self.in_channel, 3, 1, 1, activation='relu', norm='batch'))
        
        self.inplanes = self.in_channel
        self.layer1 = self._make_layer(BasicBlock, self.in_channel, 3, 2, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, self.in_channel*2, 3, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, self.in_channel*4, 3, 1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, self.in_channel*4, 3, 1, 1, 2)

        self.lastconv = nn.Sequential(
            Conv2D(self.in_channel*10, self.in_channel*4, 3, 1, 1, activation='relu', norm='batch'),
            Conv2D(self.in_channel*4, concat_feature_channel, kernel_size=1, padding=0, stride=1, activation=None, norm=None))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Conv2D(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride,padding=0,\
                activation=None, norm='batch')

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.conv_1(x)
        # x2 = self.conv_2(x1)
        l1 = self.layer1(x1)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        gwc_feature = torch.cat((l2, l3, l4), dim=1)
        concat_feature = self.lastconv(gwc_feature)
        return {"gwc_feature": gwc_feature, "concat_feature": concat_feature}

class similarity_computation(nn.Module):
    def __init__(self, maxdisp, num_groups, concat_channels):
        super(similarity_computation, self).__init__()
        self.maxdisp = maxdisp
        self.num_groups = num_groups
        self.concat_channels = concat_channels

        self.compress = nn.Sequential(
            Conv3D(self.num_groups + self.concat_channels * 2, 32, 3, 1, 1, activation='relu', norm='batch'),
            Conv3D(32, 32, 3, 1, 1, activation='relu', norm='batch'))
    
    def build_concat_volume(self, refimg_fea, targetimg_fea):
        B, C, H, W = refimg_fea.shape
        volume = refimg_fea.new_zeros([B, 2 * C, self.maxdisp, H, W])
        for i in range(self.maxdisp):
            if i > 0:
                volume[:, :C, i, :, i:] = refimg_fea[:, :, :, i:]
                volume[:, C:, i, :, i:] = targetimg_fea[:, :, :, :-i]
            else:
                volume[:, :C, i, :, :] = refimg_fea
                volume[:, C:, i, :, :] = targetimg_fea
        volume = volume.contiguous()
        return volume

    def build_gwc_volume(self, refimg_fea, targetimg_fea):
        B, C, H, W = refimg_fea.shape
        volume = refimg_fea.new_zeros([B, self.num_groups, self.maxdisp, H, W])
        for i in range(self.maxdisp):
            if i > 0:
                volume[:, :, i, :, i:] = self.groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i],
                                                            self.num_groups)
            else:
                volume[:, :, i, :, :] = self.groupwise_correlation(refimg_fea, targetimg_fea, self.num_groups)
        volume = volume.contiguous()
        return volume
    
    def groupwise_correlation(self, fea1, fea2, num_groups):
        B, C, H, W = fea1.shape
        assert C % num_groups == 0
        channels_per_group = C // num_groups
        cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
        assert cost.shape == (B, num_groups, H, W)
        return cost

    def forward(self, features_left, features_right):
        gwc_volume = self.build_gwc_volume(features_left["gwc_feature"], features_right["gwc_feature"])
        concat_volume = self.build_concat_volume(features_left["concat_feature"], features_right["concat_feature"])
        volume = torch.cat((gwc_volume, concat_volume), 1)
        volume = self.compress(volume)
        return volume

class pyramid_pooling(nn.Module):
    def __init__(self, sizes=(1, 3, 6, 8), dimension=3):
        super(pyramid_pooling, self).__init__()
        self.stages = nn.ModuleList([self._make_stage(size, dimension) for size in sizes])

    def _make_stage(self, size, dimension=3):
        if dimension == 1:
            prior = nn.AdaptiveAvgPool1d(output_size=size)
        elif dimension == 2:
            prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        elif dimension == 3:
            prior = nn.AdaptiveAvgPool3d(output_size=(size, size, size))
        return prior

    def forward(self, feats):
        n, c, _, _, _ = feats.size()
        priors = []
        for stage in self.stages:
            p = stage(feats).view(n, c, -1) # 1, 27, 216, 512
            priors.append(p)
        center = torch.cat(priors, -1) # 756
        return center

class channel_propagate(nn.Module):
    def __init__(self, in_channels, pp_size=(1, 3, 6, 8)):
        super(channel_propagate, self).__init__()
        self.to_q = Conv3D(in_channels, in_channels, 1, 1, 0, activation=None, norm='batch')
        self.to_k = Conv3D(in_channels, in_channels, 1, 1, 0, activation=None, norm='batch')
        self.to_v = Conv3D(in_channels, in_channels, 1, 1, 0, activation=None, norm='batch')
        self.pp = pyramid_pooling(pp_size)
    
    def forward(self, x):
        b, c, d, h, w = x.shape
        q = self.pp(self.to_q(x)) # b, c, n
        k = self.pp(self.to_k(x)).permute(0, 2, 1)  # b, n, c
        v = self.to_v(x).view(b, c, -1)# b, c, dhw
        affinity = torch.matmul(q, k) # b, c, c
        affinity = F.softmax(affinity, dim=-1) # b, c, c
        context = torch.matmul(affinity, v) # b, c, dhw
        context = context.view(b, c, d, h, w).contiguous()
        return context, affinity

class spatial_propagate(nn.Module):
    def __init__(self, in_channel, scale=4, pp_size=(1, 3, 6, 8)):
        super(spatial_propagate, self).__init__()
        self.channels = in_channel // scale
        self.to_q = Conv3D(in_channel, self.channels, 1, 1, 0, activation=None, norm='batch')
        self.to_k = Conv3D(in_channel, self.channels, 1, 1, 0, activation=None, norm='batch')
        self.to_v = Conv3D(in_channel, self.channels, 1, 1, 0, activation=None, norm='batch')
        self.conv = Conv3D(self.channels, in_channel, 1, 1, 0, activation=None, norm='batch')
        self.pp = pyramid_pooling(pp_size)

    def forward(self, x):
        b, c, d, h, w = x.shape

        q = self.to_q(x).view(b, self.channels, -1).permute(0, 2, 1) # b, dhw, c1
        k = self.pp(self.to_k(x)) # b, c1, n
        v = self.pp(self.to_v(x)).view(b, self.channels, -1).permute(0, 2, 1) # b, n, c2
        
        affinity = torch.matmul(q, k) # b, dhw, n
        affinity = (self.channels ** -.5) * affinity # b, dhw, n
        affinity = F.softmax(affinity, dim=-1) # b, dhw, n

        context = torch.matmul(affinity, v) # b, dhw, c2
        context = context.permute(0, 2, 1).contiguous() # b, c2, dhw
        context = context.view(b, self.channels, d, h, w) # b, c2, d, h, w
        context = self.conv(context) # b, c, d, h, w
        return context, affinity.view(b, d, h, w, -1)

class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = Conv3D(in_channels, in_channels * 2, 3, 2, 1, activation='relu', norm='batch')

        self.conv2 = Conv3D(in_channels * 2, in_channels * 2, 3, 1, 1, activation='relu', norm='batch')

        self.conv3 = Conv3D(in_channels * 2, in_channels * 4, 3, 2, 1, activation='relu', norm='batch')

        self.conv4 = Conv3D(in_channels * 4, in_channels * 4, 3, 1, 1, activation='relu', norm='batch')

        self.cp = channel_propagate(in_channels * 4)

        self.conv_cp = Conv3D(in_channels*8, in_channels*4, 1, 1, 0, activation='relu', norm='batch')

        self.sp = spatial_propagate(in_channels * 4)

        self.conv_sp = Conv3D(in_channels*8, in_channels*4, 1, 1, 0, activation='relu', norm='batch')
        
        self.conv5 = Deconv3D(in_channels * 4, in_channels * 2, 3, 2, 1, 1, activation=None, norm='batch')
 
        self.conv6 = Deconv3D(in_channels * 2, in_channels, 3, 2, 1, 1, activation=None, norm='batch')

        self.redir1 = Conv3D(in_channels, in_channels, 1, 1, 0, activation=None, norm='batch')
        self.redir2 = Conv3D(in_channels * 2, in_channels * 2, 1, 1, 0, activation=None, norm='batch')

    def forward(self, x): 
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        cp, acc = self.cp(conv4)
        conv_cp = self.conv_cp(torch.cat((conv4, cp), dim=1))
        sp, ass = self.sp(conv4) 
        conv_sp = self.conv_sp(torch.cat((conv4, sp), dim=1))

        conv5 = F.relu(self.conv5(conv_cp+conv_sp) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)

        return conv6, acc, ass

class feedback_refine(nn.Module):
    def __init__(self):
        super(feedback_refine, self).__init__()
        self.inplanes = 32

        self.l_1 = Conv2D(3, self.inplanes, 3, 1, 1, activation='relu', norm='batch')
        self.l_2 = nn.Sequential(
            ShuffleBlock(self.inplanes, self.inplanes, 2, 2),
            ShuffleBlock(self.inplanes, self.inplanes, 1, 1),
            ShuffleBlock(self.inplanes, self.inplanes*2, 2, 2),
            ShuffleBlock(self.inplanes*2, self.inplanes*2, 1, 1))

        self.d_1 = Conv2D(1, self.inplanes, 3, 1, 1, activation='relu', norm='batch')
        self.d_2 = nn.Sequential(
            ShuffleBlock(self.inplanes, self.inplanes, 2, 2),
            ShuffleBlock(self.inplanes, self.inplanes, 1, 1),
            ShuffleBlock(self.inplanes, self.inplanes*2, 2, 2),
            ShuffleBlock(self.inplanes*2, self.inplanes*2, 1, 1))
        
        # self.c_1 = nn.Sequential(
        #     Conv2D(1, self.inplanes, 3, 1, 1, activation='relu', norm='batch'),
        #     ShuffleBlock(self.inplanes, self.inplanes, 1, 1))

        self.e1 = EdgeBlock(self.inplanes)
        self.e2 = EdgeBlock(self.inplanes)

        # Channel attention
        self.se = SEBlock(self.inplanes*2)
        # nn.Sequential(
        #     SEBlock(self.inplanes*2),
        #     ShuffleBlock(self.inplanes*2, self.inplanes*2, 1, 2),
        #     nn.Sigmoid())

        self.fd1 = ShuffleBlock(self.inplanes*2, self.inplanes*2, 1, 2)
        self.fd2 = Conv2D(self.inplanes*2, self.inplanes*2, 3, 1, 1, activation='relu', norm='batch')

        self.output = nn.Sequential(
            Conv2D(self.inplanes*2, self.inplanes, 3, 1, 1, activation='relu', norm='batch'),
            Conv2D(self.inplanes, 1, 3, 1, 1, activation='relu', norm='batch'))
        
    def forward(self, left, disparity):
        l_1 = self.l_1(left)
        l_2 = self.l_2(l_1)
        d_1 = self.d_1(disparity)
        d_2 = self.d_2(d_1)
        # c_1 = self.c_1(conf)
        
        # edge
        eb1, e_rgb = self.e1(l_1, l_2)
        eb2, e_disp = self.e2(d_1, d_2)
        eb, e_f = self.se(torch.cat((eb1, eb2), dim=1))

        # feedback
        fd1 = self.fd1(torch.cat((l_1, d_1), dim=1)) + eb 
        fd2 = self.fd2(fd1) + eb
        edge = self.output(fd2)
        return edge, eb, e_rgb, e_disp, e_f, fd2

class CalNet(nn.Module):
    def __init__(self, maxdisp, num_groups = 40, concat_channels = 12):
        super(CalNet, self).__init__()
        self.maxdisp = maxdisp
        self.num_groups = num_groups
        self.concat_channels = concat_channels

        self.feature_extraction = feature_extraction(concat_feature_channel=self.concat_channels)
        parameters = filter(lambda p: p.requires_grad, self.feature_extraction.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        print('\t=> feature_extraction trainable parameters: %.4fM' % parameters)
        self.sc = similarity_computation(self.maxdisp//4, self.num_groups, self.concat_channels)
        
        self.hg1 = hourglass(32)
        self.hg2 = hourglass(32)
        self.hg3 = hourglass(32)

        parameters = filter(lambda p: p.requires_grad, self.hg1.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters])*3 / 1_000_000
        print('\t=> hg1 trainable parameters: %.4fM' % parameters)

        self.classif0 = nn.Sequential(
            Conv3D(32, 32, 3, 1, 1, activation='relu', norm='batch'),
            Conv3D(32, 4, 3, 1, 1, activation=None, norm=None))
        self.classif1 = nn.Sequential(
            Conv3D(32, 32, 3, 1, 1, activation='relu', norm='batch'),
            Conv3D(32, 4, 3, 1, 1, activation=None, norm=None))
        self.classif2 = nn.Sequential(
            Conv3D(32, 32, 3, 1, 1, activation='relu', norm='batch'),
            Conv3D(32, 4, 3, 1, 1, activation=None, norm=None))
        self.classif3 = nn.Sequential(
            Conv3D(32, 32, 3, 1, 1, activation='relu', norm='batch'),
            Conv3D(32, 4, 3, 1, 1, activation=None, norm=None))

        self.fr = feedback_refine()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, left, right, gt=None, mask=None):
        features_left = self.feature_extraction(left)
        features_right = self.feature_extraction(right)

        volume = self.sc(features_left, features_right)

        b, _, _, h, w = volume.shape
        volume1,_,_ = self.hg1(volume)
        h2,_,_ = self.hg2(volume1)
        volume2 = h2+volume
        h3,a1,a2= self.hg3(volume2)
        volume3 = h3+volume
        # volume1 = self.hg1(volume)
        # volume2 = self.hg2(volume1)+volume
        # volume3 = self.hg3(volume2)+volume
        
        volume = self.classif0(volume).permute(0, 2, 1, 3, 4).reshape(b, 1, -1, h, w).squeeze(1) #b, d, h/4, w/4
        similarity = F.interpolate(volume, [left.size()[2], left.size()[3]], mode='bilinear', align_corners=True)
        similarity = F.softmax(similarity.squeeze(1), dim=1)

        volume1 = self.classif1(volume1).permute(0, 2, 1, 3, 4).reshape(b, 1, -1, h, w).squeeze(1) #b, d, h/4, w/4
        similarity1 = F.interpolate(volume1, [left.size()[2], left.size()[3]], mode='bilinear', align_corners=True)
        similarity1 = F.softmax(similarity1.squeeze(1), dim=1)

        volume2 = self.classif2(volume2).permute(0, 2, 1, 3, 4).reshape(b, 1, -1, h, w).squeeze(1) #b, d, h/4, w/4
        similarity2 = F.interpolate(volume2, [left.size()[2], left.size()[3]], mode='bilinear', align_corners=True)
        similarity2 = F.softmax(similarity2.squeeze(1), dim=1)

        volume3 = self.classif3(volume3).permute(0, 2, 1, 3, 4).reshape(b, 1, -1, h, w).squeeze(1) #b, d, h/4, w/4
        similarity3 = F.interpolate(volume3, [left.size()[2], left.size()[3]], mode='bilinear', align_corners=True)
        similarity3 = F.softmax(similarity3.squeeze(1), dim=1)
                    
        with torch.no_grad():
            pred0 = predict_disparity(similarity, self.maxdisp)
            pred1 = predict_disparity(similarity1, self.maxdisp)
            pred2 = predict_disparity(similarity2, self.maxdisp)
            pred3 = predict_disparity(similarity3, self.maxdisp)
        
        residual3, eb, e_rgb, e_disp, e_f, fd2 = self.fr(left, pred3)
        pred4 = residual3 + pred3

        pred0 = pred0.squeeze(1)
        pred1 = pred1.squeeze(1)
        pred2 = pred2.squeeze(1)
        pred3 = pred3.squeeze(1)
        pred4 = pred4.squeeze(1)
        conf3 = similarity3.max(1).values.squeeze(1)
        if self.training:
            crossentropy= crossentropy_loss([similarity, similarity1, similarity2, similarity3], \
                [pred0, pred1, pred2, pred3], gt, mask)
            smooth_l1 = smooth_l1_loss([pred4], [conf3], pred1, gt, mask)
            return crossentropy, smooth_l1, [pred0, pred1, pred2, pred3, pred4]
        else:
            return [pred3, pred4], [residual3], similarity3, a1, a2, eb, e_rgb, e_disp, e_f, fd2
            # return [pred0], [pred0], [pred0], [pred0], [pred0], [pred0], [pred0], [pred0], [pred0], [pred0]

if __name__ == '__main__':
    import numpy as np
    l = torch.ones((1,3,256,256))
    r = torch.ones((1,3,256,256))
    gt = torch.ones((1,1,256,256))

    net = CalNet(192)

    parameters = filter(lambda p: p.requires_grad, net.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('\t==> trainable parameters: %.4fM' % parameters)
