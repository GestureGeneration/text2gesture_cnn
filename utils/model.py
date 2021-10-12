#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init(m):
    if isinstance(m, nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)


class ConvNormRelu1d(nn.Module):
    """(conv => BN => ReLU)"""
    def __init__(self, in_channels, out_channels, k, s, p):
        super(ConvNormRelu1d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=k, stride=s, padding=p),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv.apply(weights_init)

    def forward(self, x):
        x = self.conv(x)
        return x


class DoubleConv1d(nn.Module):
    """(conv => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, k, s, p):
        super(DoubleConv1d, self).__init__()
        self.block = nn.Sequential(
            ConvNormRelu1d(in_channels, out_channels, k, s, p),
            ConvNormRelu1d(out_channels, out_channels, k, s, p),
        )

    def forward(self, x):
        x = self.block(x)
        return x


class Down1d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down1d, self).__init__()
        self.block = nn.Sequential(
            # nn.MaxPool1d(2, 2),
            ConvNormRelu1d(in_channels, out_channels, k=4, s=2, p=1)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class Up1d(nn.Module):
    """Up sampling => add => Double Conv"""
    def __init__(self, in_channels, out_channels):
        super(Up1d, self).__init__()
        self.block = nn.Sequential(
            DoubleConv1d(in_channels, out_channels, k=3, s=1, p=1)
        )

    def forward(self, x, y):
        """Following the implementation in PoseGAN"""
        x = torch.repeat_interleave(x, 2, dim=2)
        x = x + y
        x = self.block(x)
        return x


class UNet1d(nn.Module):
    """
    Text Encoder
    """
    def __init__(self, in_channels, out_channels):
        super(UNet1d, self).__init__()
        self.inconv = DoubleConv1d(in_channels, out_channels, k=3, s=1, p=1)
        self.down1 = Down1d(out_channels, out_channels)
        self.down2 = Down1d(out_channels, out_channels)
        self.down3 = Down1d(out_channels, out_channels)
        self.down4 = Down1d(out_channels, out_channels)
        self.down5 = Down1d(out_channels, out_channels)
        self.up1 = Up1d(out_channels, out_channels)
        self.up2 = Up1d(out_channels, out_channels)
        self.up3 = Up1d(out_channels, out_channels)
        self.up4 = Up1d(out_channels, out_channels)
        self.up5 = Up1d(out_channels, out_channels)
        
    def forward(self, x):
        x0 = self.inconv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.up5(x, x0)
        return x


class Decoder(nn.Module):
    """
    CNN Decoder
    """
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.layers = nn.Sequential(
            DoubleConv1d(in_channels, out_channels, k=3, s=1, p=1),
            DoubleConv1d(out_channels, out_channels, k=3, s=1, p=1),
            DoubleConv1d(out_channels, out_channels, k=3, s=1, p=1),
            DoubleConv1d(out_channels, out_channels, k=3, s=1, p=1),
            nn.Conv1d(out_channels, 98, kernel_size=1, stride=1, padding=0)
        )
        self.layers.apply(weights_init)

    # x: shape = (batch, channels, frames)
    def forward(self, x):
        x = self.layers(x)
        return x


class PatchGan(nn.Module):
    """
    Motion Discriminator
        default forward input shape = (batch_size, 98, 64)
    """

    def __init__(self, in_channel=98, ndf=64):
        """
        Parameter
        ----------
        in_channel: int
            Size of input channels
        ndf: int (default=64)
            Size of feature maps in discriminator
        """
        super(PatchGan, self).__init__()
        self.layer1 = nn.Conv1d(in_channel, ndf, kernel_size=4, stride=2, padding=1)
        self.layer2 = nn.LeakyReLU(0.2, inplace=True)
        self.layer3 = ConvNormRelu1d(ndf, ndf * 2, k=4, s=2, p=1)
        self.layer4 = ConvNormRelu1d(ndf * 2, ndf * 4, k=4, s=1, p=0)
        self.layer5 = nn.Conv1d(ndf * 4, 1, kernel_size=4, stride=1, padding=0)

        self.layer1.apply(weights_init)
        self.layer5.apply(weights_init)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(F.pad(x, [1, 2], "constant", 0))
        x = self.layer5(F.pad(x, [1, 2], "constant", 0))
        return x


class UnetDecoder(nn.Module):
    """
    unet => cnn_decoder
    """
    def __init__(self, in_channels, out_channels):
        """
        Parameter
        ----------
        in_channels : int
            input channel size
        out_channels: int
            output channel size
        """
        super(UnetDecoder, self).__init__()
        self.unet = UNet1d(in_channels, out_channels)
        self.decoder = Decoder(out_channels, out_channels)

    def forward(self, x):
        x = self.unet(x)
        x = self.decoder(x)
        return x
