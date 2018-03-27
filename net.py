#

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class encapsule(nn.Module):
    def __init__(self, inp, out):
        super(encapsule, self).__init__()
        mod1 = nn.Conv2d(inp, out, 3, bias=False)
        mod2 = nn.Conv2d(out, out * 2, 3, bias=False)
        bn = nn.BatchNorm2d(out * 2)
        maxpool = nn.MaxPool2d(2, stride=2)

    def forward(x):
        x = nn.ReLU(bn(mod1(x)))
        x = nn.ReLU(bn(mod2(x)))
        x = maxpool(x)
        x = nn.ReLU(bn(mod2(x)))
        x = nn.ReLU(bn(mod2(x)))
        return x


class main_block(nn.Module):
    def __init__(self, inp, out, filters):
        super(main_block, self).__init__()
        mod1 = nn.Conv2d(inp, out, 3, bias=False)
        mod2 = nn.Conv2d(out, filters, 1, bias=False)

    def forward(x):
        x = mod1(x)
        x = mod2(x)
        return x


class conv_net(nn.Module):
    def __init__(self, inp, out, filters):
        super(conv_net, self).__init__()
        up_mod = encapsule(inp, out)
        exp_mod = main_block(out, out * 2, filters)
        down_mod = encapsule(filters, filters * 2)

        linear = nn.Linear(filters * 2, 10)

    def forward(x):
        x = up_mod(x)
        x = exp_mod(x)
        x = down_mod(x)

        return x

def zero_ref():
    return conv_net(3, 120, 120)  # in, out, filters
