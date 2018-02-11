import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelNormLayer(nn.Module):
    def __init__(self):
        super(PixelNormLayer, self).__init__()

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + 1e-8)


class WScaleLayer(nn.Module):
    def __init__(self, size):
        super(WScaleLayer, self).__init__()
        self.scale = nn.Parameter(torch.randn([1]))
        self.b = nn.Parameter(torch.randn(size))
        self.size = size

    def forward(self, x):
        x_size = x.size()
        x = x * self.scale + self.b.view(1, -1, 1, 1).expand(
            x_size[0], self.size, x_size[2], x_size[3])

        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.wscale = WScaleLayer(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = F.leaky_relu(self.wscale(x), negative_slope=0.2)
        return x


class DownscaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1):
        super(DownscaleBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.wscale = WScaleLayer(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = F.leaky_relu(self.wscale(x), negative_slope=0.2)
        return x


class NormConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(NormConvBlock, self).__init__()
        self.norm = PixelNormLayer()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.wscale = WScaleLayer(out_channels)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv(x)
        x = F.leaky_relu(self.wscale(x), negative_slope=0.2)
        return x


class NormDenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(NormDenseBlock, self).__init__()
        self.norm = PixelNormLayer()
        self.conv = nn.Linear(in_channels, out_channels, bias=False)
        self.wscale = WScaleLayer(out_channels)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv(x)
        x = self.wscale(x)
        return x


class NormUpscaleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(NormUpscaleConvBlock, self).__init__()
        self.norm = PixelNormLayer()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.wscale = WScaleLayer(out_channels)

    def forward(self, x):
        x = self.norm(x)
        x = self.up(x)
        x = self.conv(x)
        x = F.leaky_relu(self.wscale(x), negative_slope=0.2)
        return x


class StdevAug(nn.Module):
    def __init__(self):
        super(StdevAug, self).__init__()

    # Note slight deviation from original implementation, which used an offset:
    # T.sqrt(Tmean(T.square(val - Tmean(val, **kwargs)), **kwargs) + 1.0e-8)
    def forward(self, x):
        return torch.cat((x, torch.std(x, 1).view(-1, 1, 4, 4)), 1)


class UnitNorm(nn.Module):
    def __init__(self):
        super(UnitNorm, self).__init__()

    def forward(self, x):
        return x / torch.norm(x)
