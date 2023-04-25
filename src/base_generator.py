import torch.nn as nn

from .model_utils import conv, up_conv, ResnetBlock

class CycleGenerator(nn.Module):
    """Architecture of the generator network."""

    def __init__(self, conv_dim=64, init_zero_weights=False, norm='instance'):
        super().__init__()

        self.conv1 = conv(3, 32, 4, 2, 1, norm, False, 'relu')
        self.conv2 = conv(32, 64, 4, 2, 1, norm, False, 'relu')

        self.resnet_block = nn.Sequential(
            ResnetBlock(conv_dim, norm, 'relu'),
            ResnetBlock(conv_dim, norm, 'relu'),
            ResnetBlock(conv_dim, norm, 'relu')
        )

        self.up_conv1 = up_conv(64, 32, 3, 1, 1, 2, 'instance', 'relu')
        self.up_conv2 = up_conv(32, 3, 3, 1, 1, 2, '', 'tanh')


    def forward(self, x):
      
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.resnet_block(x)
        x = self.up_conv1(x)
        x = self.up_conv2(x)
        return x

