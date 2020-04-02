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
model = models.unsupervised_Discriminator(1, 3, 64)
model.load_state_dict(torch.load(args.pre_trained_net))
print(model)

print('load target data: ',args.dataset)
if args.dataset == 'mnist':
    transform = transforms.Compose([
        transforms.Scale(32),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True, transform=transform),
        batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, download=True, transform=transform),
        batch_size=128, shuffle=True)
    print("finished loading mnist")

else:
    _, test_loader = data_loader.getTargetDataSet(args.dataset, args.batch_size, args.imageSize, args.dataroot)

global nt_test_loader
if args.out_dataset == 'mnist':
    transform = transforms.Compose([
        transforms.Scale(32),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize(mean=(1.0, 1.0, 1.0), std=(1.0, 1.0, 1.0))
    ])

    print('load non target data: ',args.out_dataset)
    nt_test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, download=True, transform=transform),
        batch_size=128, shuffle=True)
    print("finished loading mnist non target")

else:
    print('load non target data: ',args.out_dataset)
    nt_test_loader = data_loader.getNonTargetDataSet(args.out_dataset, args.batch_size, args.imageSize, args.dataroot)

if args.cuda:
    model.cuda()

#norms = open('%s/norms.txt' % args.outf, 'w')

#features = list(model.features)
# print(len(features))
# https://discuss.pytorch.org/t/backward-starting-from-intermediate-layer/10189/3
#features = nn.ModuleList(features).eval()

def generate_target():
    epochNorms = 0.0
    model.eval()
    correct = 0
    total = 0
    f1 = open('%s/confidence_Base_In.txt'%args.outf, 'w')
    count = 0
    for data, target in test_loader:
        count +=1
        total += data.size(0)
        #vutils.save_image(data, '%s/target_samples.png'%args.outf, normalize=True)
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)


        out = data
        epochNorms += torch.mean((model(out)) ** 2).data.item()

    print ("generated target epoch norms", epochNorms/count)

def generate_non_target():
    epochNorms = 0.0
    model.eval()
    total = 0
    count = 0
    for data, target in nt_test_loader:
        total += data.size(0)
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        out = data
        epochNorms += torch.mean((model(out)) ** 2).data.item()
        count +=1

    print("generated non-target epoch norms", epochNorms / count)


print('generate log from in-distribution data')
generate_target()
print('generate log  from out-of-distribution data')
generate_non_target()
#print('calculate metrics')
#callog.metric(args.outf)
#!!!!!!!!!!!!tomorrow, check other out distribution datasets