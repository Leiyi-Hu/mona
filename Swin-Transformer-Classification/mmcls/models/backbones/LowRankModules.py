import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, factor=8, bias=True):
        super().__init__()

        self.rank = min(in_features // factor, out_features // factor)
        self.rank = max(1, self.rank)

        self.W_down = nn.Parameter(torch.zeros(in_features, self.rank))
        self.W_up = nn.Parameter(torch.zeros(out_features, self.rank))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, ))
        else:
            self.bias = torch.zeros(out_features)

        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_uniform_(self.W_down, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_up, a=math.sqrt(5))

    def forward(self, x):
        W = self.W_up @ self.W_down.T
        out = F.linear(x, W, self.bias)

        return out


class LowRankConv2d(nn.Module):  # note: not useful when give depth-wise conv.
    def __init__(self, in_features, out_features, padding=0, kernel_size=3, groups=1, dilation=1, stride=1, factor=8,
                 bias=True):
        super().__init__()

        self.rank = min(in_features // factor, out_features // factor)
        self.rank = max(1, self.rank)

        self.W_down = nn.Parameter(torch.zeros(self.rank * kernel_size, in_features * kernel_size))
        self.W_up = nn.Parameter(torch.zeros(out_features // groups * kernel_size, self.rank * kernel_size))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, ))
        else:
            self.bias = torch.zeros(out_features)

        self.kernel_size = kernel_size
        self.groups = groups
        self.in_features = in_features
        self.out_features = out_features
        self.padding = padding
        self.dilation = dilation
        self.stride = stride

        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_uniform_(self.W_down, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_up, a=math.sqrt(5))

    def forward(self, x):
        W = (self.W_up @ self.W_down).view(self.out_features, self.in_features // self.groups, self.kernel_size,
                                           self.kernel_size)

        out = F.conv2d(x, W, bias=self.bias, padding=self.padding, stride=self.stride, dilation=self.dilation,
                       groups=self.groups)

        return out
