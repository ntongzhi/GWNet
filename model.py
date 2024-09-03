import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from attention_model import PAM_Module
import os


def double_conv(in_channels, out_channels):
    """
    double conv
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class BasicBlock(nn.Module):
    """
    Basic Block for resnet18 or resnet34
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.res_func = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels*BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels*BasicBlock.expansion)
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.res_func(x) + self.shortcut(x))


class AttenDown(nn.Module):
    """
    Down Maxpool => doubleConv
    """
    def __init__(self, in_channels, out_channels):
        super(AttenDown, self).__init__()
        self.down_conv = nn.Sequential(
            nn.MaxPool2d(2),
            BasicBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down_conv(x)


class AttenUp(nn.Module):
    """
    up => pad => cat => conv
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(AttenUp, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)

        self.conv = BasicBlock(out_channels * 3, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.shape[2] - x1.shape[2]
        diffX = x2.shape[3] - x1.shape[3]

        x1 = F.pad(x1, [diffX//2, diffX-diffX//2, diffY//2, diffY - diffY//2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.down = AttenDown(in_channels, in_channels * 2)
        self.up = AttenUp(in_channels * 2, in_channels)
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = x
        x = self.down(x1)
        x = self.up(x, x1)
        x = self.conv1(x)
        return self.sigmoid(x)


class Up(nn.Module):
    """
    up => pad => cat => conv
    """
    def __init__(self, in_cahnnels, out_channels, up_channel, reduce, atten_model, bilinear=False):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(up_channel, up_channel, kernel_size=2, stride=2)
        self.conv = double_conv(in_cahnnels, out_channels)
        self.atten_conv = BasicBlock(up_channel, up_channel // reduce)
        self.atten = atten_model
        self.atten_last_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x1, x2, x3, atten_last):
        x1 = self.up(x1)
        diffY = x2.shape[2] - x1.shape[2]
        diffX = x2.shape[3] - x1.shape[3]
        x1 = F.pad(x1, [diffX//2, diffX-diffX//2, diffY//2, diffY - diffY//2])
        atten = self.atten(self.atten_conv(x1))
        if atten_last is not None:
            atten = torch.log2(self.atten_last_up(atten_last) + 1) * atten
        x2 = (x2 - x3) * atten.expand_as(x3)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x), atten


class SiamEncorder(nn.Module):
    def __init__(self, in_channels, block, num_block):
        super(SiamEncorder, self).__init__()
        self.resnet_name = []
        self.channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)

        self.mutual_atten = PAM_Module(512)
        # self.self_atten = PAM_Module(512)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.channels, out_channels, stride))
            self.channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward_once(self, x):
        conv1 = self.conv1(x)
        temp = self.maxpool(conv1)
        conv2 = self.conv2_x(temp)
        conv3 = self.conv3_x(conv2)
        conv4 = self.conv4_x(conv3)
        bottle = self.conv5_x(conv4)
        return bottle, conv4, conv3, conv2, conv1
        # return bottle, conv4.detach(), conv3.detach(), conv2.detach(), conv1.detach()

    def forward(self, x1, x2):
        x_shape = x1.shape
        feature_1 = self.forward_once(x1)
        feature_2 = self.forward_once(x2)
        x = self.mutual_atten(feature_1[0], feature_1[0], feature_1[0]) \
            - self.mutual_atten(feature_1[0], feature_2[0], feature_2[0])
        # x = self.self_atten(x, x, x)
        # x = x + feature_2[0]
        return x, feature_1, feature_2, x_shape

    def load_pretrained_weights(self):
        model_dict = self.state_dict()
        resnet34_dict = models.resnet34(True).state_dict()
        count_res = count_my = 1
        reskeys = list(resnet34_dict.keys())
        mykeys = list(model_dict.keys())

        corresp_map = []
        while True:
            reskey = reskeys[count_res]
            mykey = mykeys[count_my]

            if 'fc' in reskey:
                break
            while reskey.split('.')[-1] not in mykey:
                count_my += 1
                mykey = mykeys[count_my]
            corresp_map.append([reskey, mykey])
            count_res += 1
            count_my += 1
        for k_res, k_my in corresp_map:
            model_dict[k_my] = resnet34_dict[k_res]
            self.resnet_name.append(k_my)
        try:
            self.load_state_dict(model_dict)
            print('Loaded resnet34 weights in mynet')
        except:
            print('Error resnet34 weights in mynet')
            raise


class Decorder(nn.Module):
    def __init__(self, out_channels):
        super(Decorder, self).__init__()
        self.atten_model = SpatialAttention(16)
        self.dconv_up3 = Up(256 + 512, 256, 512, 32, self.atten_model)
        self.dconv_up2 = Up(128 + 256, 128, 256, 16, self.atten_model)
        self.dconv_up1 = Up(64 + 128, 64, 128, 8, self.atten_model)
        self.dconv_up0 = Up(64 + 64, 64, 64, 4, self.atten_model)

        self.upsample = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dconv_last2 = double_conv(64, 64)
        self.dconv_last = nn.Conv2d(64, out_channels, 1)

    def forward(self, x, feature_1, feature_2, x_shape):
        x, atten = self.dconv_up3(x, feature_1[1], feature_2[1], None)
        # x_visualize = atten
        # x_visualize = torch.mean(-x_visualize, dim=1, keepdim=True)
        # x_visualize = F.interpolate(x_visualize, size=(256, 256), mode='bilinear', align_corners=False)
        x, atten = self.dconv_up2(x, feature_1[2], feature_2[2], atten)
        x, atten = self.dconv_up1(x, feature_1[3], feature_2[3], atten)
        x, atten = self.dconv_up0(x, feature_1[4], feature_2[4], atten)
        x = self.upsample(x)
        diffY = x_shape[2] - x.shape[2]
        diffX = x_shape[3] - x.shape[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = self.dconv_last2(x)
        out = self.dconv_last(x)
        return out#, x_visualize


class SiamResUNet(nn.Module):
    """
    ResNet34-UnNet
    """
    def __init__(self, in_channels, out_channels, block, num_block):
        super(SiamResUNet, self).__init__()
        self.siam_encorder = SiamEncorder(in_channels, block, num_block)
        self.siam_encorder.load_pretrained_weights()
        self.decorder = Decorder(out_channels)

    def forward(self, x1, x2):
        x, feature_1, feature_2, x_shape = self.siam_encorder(x1, x2)
        out = self.decorder(x, feature_1, feature_2, x_shape)
        # out, x_visualize = self.decorder(x, feature_1, feature_2, x_shape)
        return out#, x_visualize

    def save_model(self, model_path):
        torch.save(self.siam_encorder.state_dict(), os.path.join(model_path, 'best_siam_encorder.pth'))
        torch.save(self.decorder.state_dict(), os.path.join(model_path, 'best_decorder.pth'))

    def load_model(self, model_path):
        self.siam_encorder.load_state_dict(torch.load(os.path.join(model_path, 'best_siam_encorder.pth')))
        self.decorder.load_state_dict(torch.load(os.path.join(model_path, 'best_decorder.pth')))
        print('loaded trained model')




if __name__ == '__main__':
    device = torch.device('cuda')
    net = SiamResUNet(3, 1, BasicBlock, [3, 4, 6, 3])
    net.to(device)
    x1, x2 = torch.rand((2, 3, 256, 256)), torch.rand((2, 3, 256, 256))
    x1, x2 = x1.to(device), x2.to(device)
    x = net(x1, x2)
    print(x.shape)
