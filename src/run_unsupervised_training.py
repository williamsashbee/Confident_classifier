##############################################
# This code is based on samples from pytorch #
##############################################
# Writer: Kimin Lee
from __future__ import print_function
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import data_loader
import numpy as np
import torchvision.utils as vutils
import models

from torchvision import datasets, transforms
from torch.autograd import Variable

# os.environ["CUDA_LAUNCH_BLOCKING"]="1"

# Training settings
parser = argparse.ArgumentParser(description='Training code - joint confidence')
parser.add_argument('--batch-size', type=int, default=128, help='input batch size for training')
parser.add_argument('--save-interval', type=int, default=3, help='save interval')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.00002, help='learning rate')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--log-interval', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--dataset', default='cifar10', help='mnist | cifar10 | svhn')
parser.add_argument('--dataroot', default='.', help='path to dataset')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
parser.add_argument('--droprate', type=float, default=0.1, help='learning rate decay')
parser.add_argument('--decreasing_lr', default='60', help='decreasing strategy')
parser.add_argument('--num_classes', type=int, default=10, help='the # of classes')


args = parser.parse_args()

if args.dataset == 'cifar10':
    #args.beta = 0.1
    args.batch_size = 64

print(args)
args.cuda = not args.no_cuda and torch.cuda.is_available()
print("Random Seed: ", args.seed)
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

print('load data: ', args.dataset)

mnist_mean = [0.1307, 0.1307, 0.1307]
mnist_std = [0.3081, 0.3081, 0.3081]

if args.dataset == 'mnist':
    print ("mnist is train_loader")
    transform = transforms.Compose([
        transforms.Scale(32),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize(mean=mnist_mean, std=mnist_std)
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True, transform=transform),
        batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, download=True, transform=transform),
        batch_size=128, shuffle=True)
else:
    print(args.dataset, " is train_loader")
    train_loader, test_loader = data_loader.getTargetDataSet(args.dataset, args.batch_size, args.imageSize,
                                                             args.dataroot)


print('Setup optimizer')
decreasing_lr = list(map(int, args.decreasing_lr.split(',')))

#def customized_loss(X):
#    norm = torch.norm(X,p=2,dim=0)
#    loss = torch.sum(Variable(norm, requires_grad = True))/X.shape[0]
#    return Variable(loss, requires_grad = True)


#https://discuss.pytorch.org/t/custom-loss-functions/29387
#https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
global lam
lam = 1.0

#https://stackoverflow.com/questions/44641976/in-pytorch-how-to-add-l1-regularizer-to-activations
def my_loss(D, x , target, invalue):
    global lam
    lin, outin = D(x[target==invalue])
    lossin = None
    """for a in lin:
        if lossin == None:
            lossin = torch.mean(torch.sum( a**2, dim = (1,2,3)))
        else:
            lossin += torch.mean(torch.sum( a**2, dim = (1,2,3)))
    """
    all_linear1_params = torch.cat([x.view(-1) for x in lin])
    lossin = torch.mean(all_linear1_params ** 2)
    """lout, outout = D(x[target != invalue])
    lossout = None
    for a in lout:
        if lossout == None:
            lossout = torch.mean(torch.sum( a**2, dim = (1,2,3)))
        else:
            lossout += torch.mean(torch.sum( a**2, dim = (1,2,3)))
    """
    term1 = -lossin#/lin[0].shape[0]
    #term2 = lossout#/lout[0].shape[0]


    all_linear1_params = torch.cat([x.view(-1) for x in D.parameters()])
    l1_reg = torch.norm(all_linear1_params, 1)

    #if term1 < -1000:
    #    lam *= 10.0
    #else:
    #    lam *= .1
    lam = torch.abs(term1).data.item()*.1
#    return term1 #+ .5*term2 #+ lam * l2_reg
    return term1 + lam * l1_reg

def train(epoch):
    D.train()
    returnLoss = 0.0
    returnKL = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):


        if args.cuda:
            data, target = data.cuda(), target.cuda()

        data, target =  Variable(data), Variable(target)

        ###########################
        # (1) Update D network    #
        ###########################
        # train with real
        D_optimizer.zero_grad()
        #x = torch.randn(128, 3,32,32).cuda()
        x = data

        if x[target == 0].shape[0] ==0:
            continue
        errD_real = my_loss(D,x,target, invalue=0)
        errD_real.retain_grad()
        errD_real.backward()
        D_optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Classification Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), errD_real.data.item()))
            print('errD_real', errD_real.grad)
            #print('sum',torch.sum(D))

    return

"""
def test(epoch):
    netD.eval()
    test_loss = 0
    correct = 0
    total = 0
    for data, target in test_loader:
        total += data.size(0)
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = F.log_softmax(netD(data))
        test_loss += F.nll_loss(output, target.type(torch.cuda.LongTensor).reshape((target.shape[0],))).data.item()
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.type(torch.cuda.LongTensor).reshape((target.shape[0],)).data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(test_loader)  # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, total,
        100. * correct / total))

"""


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


global D
global D_optimizer


while True:
    #new_classifier = nn.Sequential(*list(model.classifier.children())[:-1])https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3
    D = Vgg13()   # ngpu, nc, ndf

    if args.cuda:
        D.cuda()
    D_optimizer = optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    for epoch in range(1, args.epochs + 1):
        train(epoch)

#        test(epoch)
        if epoch in decreasing_lr:
            D_optimizer.param_groups[0]['lr'] *= args.droprate
        if epoch % 5 == 0:
            # do checkpointing

            torch.save(D.state_dict(),
                       '%s/%s-%s_netD%03d.pth' % (
                           args.outf, "unsupervised_discriminator.pth", args.dataset, epoch))

#https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#sphx-glr-beginner-transfer-learning-tutorial-py
#https://stackoverflow.com/questions/55083642/extract-features-from-last-hidden-layer-pytorch-resnet18
