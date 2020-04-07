###################################################################################################
# Measure the detection performance: reference code is https://github.com/ShiyuLiang/odin-pytorch #
###################################################################################################
# Writer: Kimin Lee
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import data_loader
import numpy as np
import torchvision.utils as vutils
import calculate_log as callog
import models
import math

#from torch.utils.serialization import load_lua
from torchvision import datasets, transforms
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from numpy.linalg import inv

# Training settings
parser = argparse.ArgumentParser(description='Test code - measure the detection peformance')
parser.add_argument('--batch-size', type=int, default=128, help='batch size')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA')
parser.add_argument('--seed', type=int, default=1,help='random seed')
parser.add_argument('--dataset', required=True, help='target dataset: cifar10 | svhn')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--outf', default='/home/rack/KM/2017_Codes/overconfidence/test/log_entropy', help='folder to output images and model checkpoints')
parser.add_argument('--out_dataset', required=True, help='out-of-dist dataset: cifar10 | svhn | imagenet | lsun')
parser.add_argument('--num_classes', type=int, default=10, help='number of classes (default: 10)')
parser.add_argument('--pre_trained_net', default='', help="path to pre trained_net")

args = parser.parse_args()
print(args)
args.cuda = not args.no_cuda and torch.cuda.is_available()
print("Random Seed: ", args.seed)
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

print('Load model')

class Vgg13(torch.nn.Module):
    def __init__(self):
        super(Vgg13, self).__init__()
        features = list(models.vgg13().features)
        self.features = nn.ModuleList(features).eval()

    def forward(self, x):
        results = []
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {24}:
                results.append(x)
        return results, x


model = Vgg13()
model.load_state_dict(torch.load(args.pre_trained_net))
print(model)

cifar10_train_loader, cifar10_test_loader = data_loader.getTargetDataSet("cifar10", args.batch_size, args.imageSize, args.dataroot)

#cifar10_train_loader = data_loader.getTargetDataSet("cifar10", args.batch_size, args.imageSize, args.dataroot)

#cifar10_test_loader = data_loader.getNonTargetDataSet("cifar10", args.batch_size, args.imageSize, args.dataroot)

svhn_test_loader = data_loader.getNonTargetDataSet("svhn", args.batch_size, args.imageSize, args.dataroot)

stl10_test_loader = data_loader.getNonTargetDataSet("stl10", args.batch_size, args.imageSize, args.dataroot)

mnist_mean = [0.1307, 0.1307, 0.1307]
mnist_std = [0.3081, 0.3081, 0.3081]

mnist_transform = transforms.Compose([
    transforms.Scale(32),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize(mean = mnist_mean, std = mnist_std)
])


mnist_test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, download=True, transform=mnist_transform),
    batch_size=128, shuffle=True)

if args.cuda:
    model.cuda()

#norms = open('%s/norms.txt' % args.outf, 'w')

#features = list(model.features)
# print(len(features))
# https://discuss.pytorch.org/t/backward-starting-from-intermediate-layer/10189/3
#features = nn.ModuleList(features).eval()


def generate_svhn():
    totalout = 0.0
    count = 0.0
    for data, target in svhn_test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        outloss = my_loss(model, data)
        totalout += outloss
        count += 1.0
        #print("svhn out \n", outloss)
    print("svhn out \n", totalout / count)


def my_loss(D, x):
    l, out = D(x)
    total = 0.0
    loss = Variable(torch.zeros(x.shape[0],).cuda())
    loss = None
    for a in l:
        if loss == None:
            loss = torch.mean(torch.sum(a**2, dim = (1,2,3)))
        else:
            loss += torch.mean(torch.sum(a**2, dim = (1,2,3)))

    return loss

def my_losses(D, x):
    l, out = D(x)
    total = 0.0
    loss = Variable(torch.zeros(x.shape[0],).cuda())
    loss = None
    a = []
    for el in l:
        if loss == None:
            a+= torch.sum(
                    el**2,
                    dim = (1,2,3)
                ).tolist()


    return a

def plot(input, title):
    # Import the libraries
    import matplotlib.pyplot as plt
    #import seaborn as sns
    #input =  [21,22,23,4,5,6,77,8,9,10,31,32,33,34,35,36,37,18,49,50,100]
    num_bins = 40
    # matplotlib histogram
    n,bins, patches = plt.hist(input, num_bins, facecolor='blue', alpha = 0.5)
    plt.title( title )

    plt.show()

def getCifar10InOutValues():
    il = []
    ol = []

    for data, target in cifar10_train_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        inloss = my_losses(model, data[target == 0])
        il += inloss
        outloss = my_losses(model,data[target != 0])
        ol += outloss

    return il,ol

def getSVHNValues():
    ol = []

    for data, target in svhn_test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        outloss = my_losses(model,data)
        ol += outloss

    return ol


def getStl10Values():
    ol = []

    for data, target in stl10_test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        outloss = my_losses(model,data)
        ol += outloss

    return ol


def getMnistValues():
    ol = []

    for data, target in mnist_test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        outloss = my_losses(model,data)
        ol += outloss

    return ol



i, o = getCifar10InOutValues()
plot(i, 'cifar10 class 0 ')
plot(o, 'cifar10 class != 0 ')

o = getSVHNValues()
plot(o, 'svhn values')

o = getStl10Values()
plot(o, 'stl10 values')

o = getMnistValues()
plot(o, 'mnist values')
#https://discuss.pytorch.org/t/output-from-hidden-layers/6325/2