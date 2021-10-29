from config import cfg
import torch
import torch.nn as nn
import torch.nn.functional as F
from misc.layer import Conv2d, FC
from torchvision import models
from misc.utils import *

model_path = 'E:/date/vgg16-397923af.pth'

class HRANet(nn.Module):
    def __init__(self, load_weights=False):
        super(HRANet, self).__init__()
        self.layer3_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256]
        self.layer4_feat = ['M', 512, 512, 512]
        self.layer5_feat = ['M', 512, 512, 512]
        self.layer3 = make_layers(self.layer3_feat)
        self.layer4 = make_layers(self.layer4_feat,in_channels=256)
        self.layer5 = make_layers(self.layer5_feat,in_channels=512)
        self.backend_feat2 = [128]
        self.backend_feat3 = [64]
        self.backend2 = make_layers(self.backend_feat2, in_channels=256)
        self.backend3 = make_layers(self.backend_feat3, in_channels=128)
        self.output_layer = nn.Sequential(nn.Conv2d(64, 1, kernel_size=1), nn.ReLU())
        self.raam3 = RAAM(256,256)
        self.raam4 = RAAM(512,512)
        self.raam5 = RAAM(512,512)
        self.decoder4 = nn.Conv2d(512*2, 512, kernel_size=1)
        self.decoder3 = nn.Conv2d(768, 256, kernel_size=1)

        if not load_weights:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
            for i in range(len(self.layer3.state_dict().items())):
                list(self.layer3.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]
            for i in range(len(self.layer4.state_dict().items())):
                list(self.layer4.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i + 14][1].data[:]
            for i in range(len(self.layer5.state_dict().items())):
                list(self.layer5.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i + 20][1].data[:]
    def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):

        F3 = self.layer3(x)
        R3 = self.raam3(F3)
        F4 = self.layer4(F3)
        R4 = self.raam4(F4)
        F5 = self.layer5(F4)
        R5 = self.raam5(F5)
        R5 = nn.functional.interpolate(R5, scale_factor=2)
        R4 = self.decoder4(torch.cat((R4, R5), 1))
        R4= nn.functional.interpolate(R4, scale_factor=2)
        R3 = self.decoder3(torch.cat((R4, R3), 1))
        x = self.backend2(R3)
        x = nn.functional.interpolate(x, scale_factor=2)
        x = self.backend3(x)
        x = nn.functional.interpolate(x, scale_factor=2)
        x = self.output_layer(x)
        return x

def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, NL='relu', same_padding=False, bn=True, bias=True):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) // 2) if same_padding else 0

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias)

        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        if NL == 'relu' :
            self.relu = nn.ReLU(inplace=True)
        elif NL == 'prelu':
            self.relu = nn.PReLU()
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class RAM(nn.Module):
    def __init__(self, k, channel):
        super(RAM, self).__init__()
        self.k = k
        self.channel = channel
        self.conv = nn.Conv2d(channel, channel // 4, 1, padding=0, bias=True)
        self.fuse = nn.Conv2d(channel // 4, channel, 1, padding=0, bias=True)
        self.dw_conv = nn.Conv2d(channel // 4, channel // 4, self.k, padding=(self.k-1) // 2, groups=channel // 4)
        self.pool = nn.AdaptiveAvgPool2d(k)

    def forward(self, x):
        N, C, H, W = x.shape
        # [N * C/4 * H * W]
        f = self.conv(x)
        # [N * C/4 * K * K]
        g = self.conv(self.pool(x))

        f_list = torch.split(f, 1, 0)
        g_list = torch.split(g, 1, 0)

        out = []
        for i in range(N):
            #[1* C/4 * H * W]
            f_one = f_list[i]
            # [C/4 * 1 * K * K]
            g_one = g_list[i].squeeze(0).unsqueeze(1)
            self.dw_conv.weight = nn.Parameter(g_one)

            # [1* C/4 * H * W]
            o = self.dw_conv(f_one)
            out.append(o)

        # [N * C/4 * H * W]
        y = torch.cat(out, dim=0)
        y = self.fuse(y)

        return y

class RAAM(nn.Module):
    def __init__(self, features, out_features):
        super(RAAM, self).__init__()
        self.bottleneck = nn.Conv2d(features * 2, out_features, kernel_size=1)
        self.relu = nn.ReLU()
        self.weight_net = nn.Conv2d(features,features,kernel_size=1)
        self.RAM = RAM(3,out_features)

    def __make_weight(self,feature,region_feature):
        weight_feature = feature - region_feature
        return F.sigmoid(self.weight_net(weight_feature))

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        region_feature = self.RAM(feats)
        weights = self.__make_weight(feats,region_feature)
        features = region_feature * weights
        bottle = self.bottleneck(torch.cat((features,feats), 1))
        return self.relu(bottle)
