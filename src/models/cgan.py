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

class _netCD(nn.Module):
    def __init__(self, ngpu, nc, ndf):
        super(_netCD, self).__init__()
        self.ngpu = ngpu
        self.conv1_1 = nn.Conv2d(nc, 64, 4, 2, 1, bias=False)
        self.conv1_2 = nn.Conv2d(10, 64, 4, 2, 1)
        self.main = nn.Sequential(
            # input size. (nc) x 32 x 32
     #       Print(),
            nn.Conv2d(64*2, 64*4, 4, 2, 1, bias=False),
    #        Print(),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
    #        Print(),
            nn.Conv2d(64*4, 64*8, 4, 2, 1, bias=False),
   #         Print(),

            nn.BatchNorm2d(64*8),
  #          Print(),

            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
 #           Print(),

            nn.Conv2d(64*8, 64 * 16, 4, 2, 1, bias=False),
 #           Print(),

            nn.BatchNorm2d(64 * 16),
 #           Print(),

            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
#            Print("LR"),

            nn.Conv2d(64 * 16, 1, 2, 1, 0, bias=False),
#            Print("2d"),

            nn.Sigmoid()#,
 #           Print("sigmoids")

        )

    def forward(self, input, labels):

        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            x = F.leaky_relu(self.conv1_1(input), 0.2)
            y = F.leaky_relu(self.conv1_2(labels), 0.2)
            x = torch.cat([x,y],1)
            output = self.main(x)

        return output.view(-1, 1)

class _netCG(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc):
        super(_netCG, self).__init__()
        self.ngpu = ngpu
        self.deconv1_1 = nn.ConvTranspose2d(100, 128 * 2, 4, 1, 0)
        self.deconv1_1_bn = nn.BatchNorm2d(128 * 2)
        self.deconv1_2 = nn.ConvTranspose2d(10, 128 * 2, 4, 1, 0)
        self.deconv1_2_bn = nn.BatchNorm2d(128 * 2)

        self.main = nn.Sequential(
            # input is Z, going into a convolution
#            Print("generator"),

            nn.ConvTranspose2d(512, ngf , 4, 2, 1, bias=False),

            nn.BatchNorm2d(ngf ),

            nn.ReLU(True),
            #Print('2d'),

            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf , ngf/2 , 4, 2, 1, bias=False),
            #Print(),

            nn.BatchNorm2d(ngf /2),
            nn.ReLU(True),
            #Print("relu"),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf /2, 3, 4, 2, 1, bias=False),
            #Print("2d"),
            #nn.BatchNorm2d(ngf * 2),
            #nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            #nn.ConvTranspose2d(ngf * 2, nc, 4, stride = 2, padding =  1, bias=False),

            nn.Sigmoid()#,
            #Print("sigmoid")
            # state size. (nc) x 32 x 32
        )

    def forward(self, input,label):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            x = F.relu(self.deconv1_1_bn(self.deconv1_1(input)))
            y = F.relu(self.deconv1_2_bn(self.deconv1_2(label)))
            x = torch.cat([x, y], 1)

            output = self.main(x)
        return output

def cGenerator(n_gpu, nz, ngf, nc):
    model = _netCG(n_gpu, nz, ngf, nc)
    model.apply(cweights_init)
    return model

def cDiscriminator(n_gpu, nc, ndf):
    model = _netCD(n_gpu, nc, ndf)
    model.apply(cweights_init)
    return model

