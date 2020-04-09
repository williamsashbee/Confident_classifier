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


model = models.vgg13()
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
    out, features = D(x)
    total = 0.0
    loss = None
    if loss == None:
        loss = torch.sum(features**2, dim = (1,2,3))
    else:
        loss += torch.sum(features**2, dim = (1,2,3))
    return loss

def my_losses(D, x):

    out, features = D(x,False)

    a = torch.mean(
            features**2,
            dim = (1,2,3)
        ).tolist()
    assert len(a) == x.shape[0]
    return a

def plot(input, title, median):
    # Import the libraries
    import matplotlib.pyplot as plt
    #import seaborn as sns
    #input =  [21,22,23,4,5,6,77,8,9,10,31,32,33,34,35,36,37,18,49,50,100]
    num_bins = 40
    # matplotlib histogram
    #n,bins, patches = plt.hist(input, num_bins, facecolor='blue', alpha = 0.5)

    fig, (ax) = plt.subplots(1, 1, figsize=(8, 6))
    ax.hist(input, num_bins, facecolor='blue', alpha=0.5)
    ax.axvline(x=median, color='r', linestyle='dashed', linewidth=2)
    plt.title( title + "\n\nmedian: " + str(median) )
    plt.show()


def evaluateWeights(D = None):
    all_linear1_params = torch.cat([x.view(-1) for x in D.parameters()])
    norm = torch.mean(all_linear1_params**2)
    print("l2 norm of weights", norm)
    plot(all_linear1_params.tolist(),"weights hist", torch.median(all_linear1_params).data.item())


def getCifar10():
    l = []
    for data, target in cifar10_test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        inloss = my_losses(model, data)
        l += inloss

    return l, torch.median(torch.FloatTensor(l)).data.item()


def getSVHNValues():
    l = []

    for data, target in svhn_test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        outloss = my_losses(model,data)
        l += outloss

    return l, torch.median(torch.FloatTensor(l)).data.item()


def getStl10Values():
    l = []

    for data, target in stl10_test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        outloss = my_losses(model,data)
        l += outloss

    return l, torch.median(torch.FloatTensor(l)).data.item()


def getMnistValues():
    l = []

    for data, target in mnist_test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        outloss = my_losses(model,data)
        l += outloss

    return l, torch.median(torch.FloatTensor(l)).data.item()


evaluateWeights(model)

l, median = getCifar10()
plot(l, 'cifar10 norms per input', median)

l, median = getSVHNValues()
plot(l, 'svhn norms per input', median)

l, median = getStl10Values()
plot(l, 'stl10 norms per input', median)

l, median = getMnistValues()
plot(l, 'mnist norms per input', median)
#https://discuss.pytorch.org/t/output-from-hidden-layers/6325/2

