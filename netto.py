#

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class encapsule(nn.Module):
    def __init__(self, inp, out):
        super(encapsule, self).__init__()
        self.double = out*2
        self.conv1 = nn.Conv2d(inp, out, 3, bias=False)
        self.conv2 = nn.Conv2d(out, out, 3, bias=False)
        self.conv3 = nn.Conv2d(out, self.double, 3, bias= False)
        self.conv4 = nn.Conv2d(self.double, self.double, 3, bias= False)

        self.bn12 = nn.BatchNorm2d(out)
        self.bn34 = nn.BatchNorm2d(self.double)

        # self.maxpool = nn.MaxPool2d(2,stride=1)
    def forward(self, x):
        x = F.relu(self.bn12(self.conv1(x))) #
        x = F.relu(self.bn12(self.conv2(x)))
        # x = self.maxpool(x)
        x = F.relu(self.bn34(self.conv3(x)))
        x = F.relu(self.bn34(self.conv4(x)))
        return x


class main_block(nn.Module):
    def __init__(self, inp, out, filters):
        super(main_block, self).__init__()
        self.mod1 = nn.Conv2d(inp, out, 3, bias=False)
        self.mod2 = nn.Conv2d(out, filters, 1, bias=False)

    def forward(self, x):
        x = self.mod1(x)
        x = self.mod2(x)
        return x


class conv_net(nn.Module):
    def __init__(self, inp, out, filters):
        super(conv_net, self).__init__()
        self.up_mod = encapsule(inp, out)
        self.exp_mod = main_block(out*2, out*4, filters)
        self.down_mod = encapsule(filters, filters)

        self.linear = nn.Linear(filters*2, 10)

    def forward(self, x):
        x = self.up_mod(x)
        x = self.exp_mod(x)
        x = self.down_mod(x)
        x = self.linear(x)

        return x

def zero_ref():
    return conv_net(3, 120, 240) #240=out*2, while inp to conv1x1 is 480 # in, out, filters
