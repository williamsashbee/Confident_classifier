## reference code is https://github.com/pytorch/examples/blob/master/dcgan/main.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

#from models import *


def cweights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Print(nn.Module):
    def __init__(self,s=''):
        super(Print, self).__init__()
        self.msg = s
    def forward(self, x):
        print(x.shape, self.msg)
        return x



def cGenerator(n_gpu, nz, ngf, nc):
    model = generator()
    #model.apply(cweights_init)
    return model

def cDiscriminator(n_gpu, nc, ndf):
    model = discriminator()
    #model.apply(cweights_init)
    return model

class generator(nn.Module):
    # initializers
    def __init__(self, d=126):
        super(generator, self).__init__()
        self.deconv1_1 = nn.ConvTranspose2d(100, d, 4, 1, 0)
        self.deconv1_1_bn = nn.BatchNorm2d(d)
        self.deconv1_2 = nn.ConvTranspose2d(10, d, 4, 1, 0)
        self.deconv1_2_bn = nn.BatchNorm2d(d)
        self.deconv2 = nn.ConvTranspose2d(d*2, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*8, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*8)
        self.deconv4 = nn.ConvTranspose2d(d*8, 3, 4, 2, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        x = F.relu(self.deconv1_1_bn(self.deconv1_1(input)))
        y = F.relu(self.deconv1_2_bn(self.deconv1_2(label)))
        x = torch.cat([x, y], 1)
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.tanh(self.deconv4(x))
        # x = F.relu(self.deconv4_bn(self.deconv4(x)))
        # x = F.tanh(self.deconv5(x))

        return x

class discriminator(nn.Module):
    # initializers
    def __init__(self, d=226):
        super(discriminator, self).__init__()
        self.d = d
        self.conv1_1 = nn.Conv2d(3, d/2, 4, 2, 1)
        self.conv1_2 = nn.Conv2d(5, d/2, 4, 2, 1)
        self.conv2 = nn.Conv2d(d/2 , d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2+ 5, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d * 4, 1, 4, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        x = F.leaky_relu(self.conv1_1(input), 0.2)
        #y = F.leaky_relu(self.conv1_2(label), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)

        x = torch.cat([x, label], 1)

        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.sigmoid(self.conv4(x))

        return x

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
