import torch
from torch import nn
import math
from CONFIG import *

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.residual_block = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1), #Conv1
            nn.BatchNorm2d(channels), #Bn1
            nn.PReLU(), #PRELU
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1), #Conv2
            nn.BatchNorm2d(channels) #Bn2
        )

    def forward(self, x):
        return x + self.residual_block(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBlock, self).__init__()
        self.upsample_block = nn.Sequential(
            nn.Conv2d(in_channels= in_channels, out_channels=in_channels*up_scale**2, kernel_size=3, padding=1),
            nn.PixelShuffle(up_scale),
            nn.PReLU()
        )

    def forward(self, x):
        return self.upsample_block(x)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.residual_blocks = nn.Sequential(
            # ResidualBlock(64),
            # ResidualBlock(64),
            # ResidualBlock(64),
            # ResidualBlock(64),
            # ResidualBlock(64)
        )
        for i in range(5):
            self.residual_blocks.add_module('residual'+str(i), ResidualBlock(64))
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        self.upsample_blocks = nn.Sequential(
            UpsampleBlock(64,2),
            UpsampleBlock(64,2)
        )
        self.conv3 = nn.Conv2d(64, 3, kernel_size=9, padding=4)

    def forward(self, x):
        x_conv1 = self.conv1(x)
        x_residual_blocks = self.residual_blocks(x_conv1)
        x_conv2 = self.conv2(x_residual_blocks)
        x_upsample_blocks = self.upsample_blocks(x_conv1 + x_conv2)
        x_conv3 = self.conv3(x_upsample_blocks)
        return (torch.tanh(x_conv3)+1)/2


class ResidualBlocks2(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResidualBlocks2, self).__init__()
        self.residual_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.residual_block_2(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )
        self.conv_block = nn.Sequential(
            ResidualBlocks2(64, 64, 2),
            ResidualBlocks2(64, 128, 1),
            ResidualBlocks2(128, 128, 2),
            ResidualBlocks2(128, 256, 1),
            ResidualBlocks2(256, 256, 2),
            ResidualBlocks2(256, 512, 1),
            ResidualBlocks2(512, 512, 2),
            )

        self.fc = nn.Sequential(
            nn.Linear(in_features=512*(math.ceil(HR_CROP_SIZE/16))**2, out_features=1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv_block(x)
        x = x.flatten(start_dim = 1)
        return torch.sigmoid(self.fc(x))
