import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from IPython.display import clear_output

from torch.utils.data import Dataset

# RESIDUAL BLOCK

class ResidualBlock(nn.Module):
    def __init__(self, in_channel):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(
                1
            ),
            nn.Conv2d(
                in_channel, in_channel, 3
            ),
            nn.InstanceNorm2d(in_channel),
            nn.ReLU(inplace = True),
            nn.ReflectionPad2d(
                1
            ),
            nn.Conv2d(
                in_channel, in_channel, 3
            ),
            nn.InstanceNorm2d(in_channel)
        )

    # We have to define our custom Residual Block
    def forward(self, x):
        return x + self.block(x)
    


# GENERATOR BLOCK

class Generator(nn.Module):
    def __init__(self, input_shape, num_resi_blocks):
        super(Generator, self).__init__()

        channels = input_shape[0]

        out_channels = 64

        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_channels, kernel_size = 7),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace = True),
        ]
        in_channels = out_channels

        # Downsampling
        for _ in range(2):
            out_channels *= 2
            model += [
                nn.Conv2d(
                    in_channels, 
                    out_channels, 
                    kernel_size = 3, 
                    stride = 2,
                    padding = 1
                ),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace = True)
            ]
            in_channels = out_channels

        # Residual block
        for _ in range(num_resi_blocks):
            model += [ResidualBlock(out_channels)]

        # Upsampling
        for _ in range(2):
            out_channels //= 2
            model += [
                nn.Upsample(scale_factor = 2),
                nn.Conv2d(
                    in_channels = in_channels,
                    out_channels = out_channels,
                    kernel_size = 3,
                    stride = 1,
                    padding = 1
                ),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace = True)
            ]
            in_channels = out_channels

        model += [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(
                out_channels = out_channels,
                in_channels = channels,
                kernel_size = 7
            ),
            nn.Tanh(),
        ]

    def forward(self, x):
        return self.model(x)
        
# DISCRIMINATOR BLOCK

class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        