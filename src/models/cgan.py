## reference code is https://github.com/pytorch/examples/blob/master/dcgan/main.py

import torch
import torch.nn as nn
import torch.nn.functional as F

import os

#from models import *

num_labels = 10
img_size = 32
num_layers = 6
fill_list = [torch.zeros([num_labels, num_labels, img_size/int(2**x) , img_size/int(2**x) ]).cuda() for x in range(num_layers)]

for j in range (num_layers):
    for i in range(num_labels):
        fill_list[j][i, i, :, :] = 1
        assert fill_list[j][i, i, :, :].sum() == (img_size/2**j) ** 2
    # assert y_fill_.sum() == (img_size ) ** 2 * mini_batch



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

class generator(nn.Module): #https://machinelearningmastery.com/how-to-develop-an-auxiliary-classifier-gan-ac-gan-from-scratch-with-keras/
    # initializers
    def __init__(self, d=126):
        super(generator, self).__init__()
        self.deconv1_1 = nn.ConvTranspose2d(110, d*4, 4, 1, 0)
        self.deconv1_1_bn = nn.BatchNorm2d(d*4)
        self.deconv2_1 = nn.ConvTranspose2d(d*4+10, d*2, 4, 2, 1)
        self.deconv2_1_bn = nn.BatchNorm2d(d*2)
        self.deconv3_1 = nn.ConvTranspose2d(d * 2+10, d, 4, 2, 1)
        self.deconv3_1_bn = nn.BatchNorm2d(d)

        self.deconv4 = nn.ConvTranspose2d(d+10, 3, 4, 2, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        y = fill_list[5][label.squeeze().tolist()]
        assert y.shape[1] == 10
        x = torch.cat([input, y], 1)
        x = F.relu(self.deconv1_1_bn(self.deconv1_1(x)))

        x = torch.cat([x, fill_list[3][label]], 1)
        x = F.relu(self.deconv2_1_bn(self.deconv2_1(x)))

        x = torch.cat([x, fill_list[2][label]], 1)
        x = F.relu(self.deconv3_1_bn(self.deconv3_1(x)))

        x = torch.cat([x, fill_list[1][label]], 1)
        x = F.tanh(self.deconv4(x))
        # x = F.relu(self.deconv4_bn(self.deconv4(x)))
        # x = F.tanh(self.deconv5(x))

        return x

class discriminator(nn.Module):
    # initializers
    def __init__(self, d=80,num_classes = 10):
        super(discriminator, self).__init__()
        self.d = d
        self.conv1 = nn.Conv2d(3+num_classes, d, 4, 2, 1)
        self.conv1_bn = nn.BatchNorm2d(d)
        self.conv2 = nn.Conv2d(d+num_classes, 2*d, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(2*d)
        self.conv3 = nn.Conv2d(2*d+num_classes, 4*d, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(4*d)
        self.conv4 = nn.Conv2d(4*d +num_classes, 1, 4, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        x = torch.cat([input, fill_list[0][label.squeeze().tolist()]], 1)
        x = F.leaky_relu(self.conv1_bn(self.conv1(x)), 0.2)

        x = torch.cat([x, fill_list[1][label.squeeze().tolist()]], 1)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)

        x = torch.cat([x, fill_list[2][label.squeeze().tolist()]], 1)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)

        x = torch.cat([x, fill_list[3][label.squeeze().tolist()]], 1)
        x = F.sigmoid(self.conv4(x))

        return x

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
