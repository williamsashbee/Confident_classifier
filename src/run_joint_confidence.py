##############################################
# This code is based on samples from pytorch #
##############################################
# Writer: Kimin Lee 
from __future__ import print_function
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

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
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
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
parser.add_argument('--beta', type=float, default=1.0, help='penalty parameter for KL term')
parser.add_argument('--out_dataset', default="svhn", help='out-of-dist dataset: cifar10 | svhn | imagenet | lsun')

parser.add_argument('--pre_trained_net', default='', help="path to pre trained_net")

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
else:
    train_loader, test_loader = data_loader.getTargetDataSet(args.dataset, args.batch_size, args.imageSize,
                                                             args.dataroot)

print('Load model')
model = models.vgg13()
print(model)

print('load GAN')
nz = 100
netG = models.Generator(1, nz, 64, 3)  # ngpu, nz, ngf, nc
netD = models.Discriminator(1, 3, 64)  # ngpu, nc, ndf
# Initial setup for GAN
real_label = 1
fake_label = 0
criterion = nn.BCELoss()
fixed_noise = torch.FloatTensor(64, nz, 1, 1).normal_(0, 1)

if args.cuda:
    model.cuda()
    netD.cuda()
    netG.cuda()
    criterion.cuda()
    fixed_noise = fixed_noise.cuda()
fixed_noise = Variable(fixed_noise)

print('Setup optimizer')
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))
decreasing_lr = list(map(int, args.decreasing_lr.split(',')))


def train(epoch):
    model.train()
    returnLoss = 0.0
    returnKL = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):

        gan_target = torch.FloatTensor(target.size()).fill_(0)
        uniform_dist = torch.Tensor(data.size(0), args.num_classes).fill_((1. / args.num_classes))

        if args.cuda:
            data, target = data.cuda(), target.cuda()
            gan_target, uniform_dist = gan_target.cuda(), uniform_dist.cuda()

        data, target, uniform_dist = Variable(data), Variable(target), Variable(uniform_dist)

        ###########################
        # (1) Update D network    #
        ###########################
        # train with real
        gan_target.fill_(real_label)
        targetv = Variable(gan_target)
        optimizerD.zero_grad()
        output = netD(data)
        errD_real = criterion(output, targetv)
        errD_real.backward()
        D_x = output.data.mean()

        # train with fake
        noise = torch.FloatTensor(data.size(0), nz, 1, 1).normal_(0, 1).cuda()
        if args.cuda:
            noise = noise.cuda()
        noise = Variable(noise)
        fake = netG(noise)
        targetv = Variable(gan_target.fill_(fake_label))
        output = netD(fake.detach())
        errD_fake = criterion(output, targetv)
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake
        optimizerD.step()

        ###########################
        # (2) Update G network    #
        ###########################
        optimizerG.zero_grad()
        # Original GAN loss
        targetv = Variable(gan_target.fill_(real_label))
        output = netD(fake)
        errG = criterion(output, targetv)
        D_G_z2 = output.data.mean()

        # minimize the true distribution
        KL_fake_output = F.log_softmax(model(fake))
        errG_KL = F.kl_div(KL_fake_output, uniform_dist) * args.num_classes
        generator_loss = errG + args.beta * errG_KL
        generator_loss.backward()
        optimizerG.step()

        ###########################
        # (3) Update classifier   #
        ###########################
        # cross entropy loss
        optimizer.zero_grad()
        output = F.log_softmax(model(data))
        loss = F.nll_loss(output, target.type(torch.cuda.LongTensor).reshape((target.shape[0],)))

        # KL divergence
        noise = torch.FloatTensor(data.size(0), nz, 1, 1).normal_(0, 1).cuda()
        if args.cuda:
            noise = noise.cuda()
        noise = Variable(noise)
        fake = netG(noise)
        KL_fake_output = F.log_softmax(model(fake))
        KL_loss_fake = F.kl_div(KL_fake_output, uniform_dist) * args.num_classes
        total_loss = loss + args.beta * KL_loss_fake
        total_loss.backward()
        optimizer.step()

        returnLoss, returnKL = loss.data.item(), KL_loss_fake.data.item()

        if batch_idx % args.log_interval == 0:
            print('Classification Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, KL fake Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data.item(), KL_loss_fake.data.item()))
            fake = netG(fixed_noise)
            vutils.save_image(fake.data, '%s/%s-%s-%s-%s-_epoch_%03d.png' % (
            args.outf, "gan", args.dataset, args.beta, args.beta, epoch), normalize=True)

    return returnLoss, returnKL


def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    for data, target in test_loader:
        total += data.size(0)
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = F.log_softmax(model(data))
        test_loss += F.nll_loss(output, target.type(torch.cuda.LongTensor).reshape((target.shape[0],))).data.item()
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.type(torch.cuda.LongTensor).reshape((target.shape[0],)).data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(test_loader)  # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, total,
        100. * correct / total))


maxDict = {'fpr': 0.0, 'auroc': 0.0, 'error': 0.0, 'auprin': 0.0, 'auprout': 0.0}
import random
from test_detection import generate_non_target
from test_detection import generate_target
import calculate_log as callog

badBetas = open('%s/badBetas.txt' % args.outf, 'w')
nt_test_loader = data_loader.getNonTargetDataSet(args.out_dataset, args.batch_size, args.imageSize, args.dataroot)

activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook

while True:
    losscounter = 0

    args.beta = random.uniform(0, 30)
    print('betas', args.beta)
    print('Load model')
    global model
    model = models.vgg13()
    hooks = {}
    #for name, module in model.named_modules():
    #    print(name, module)
    layers = list(model.features.children())
    print(len(layers))


    model.classifier.__getitem__(6).register_forward_hook(get_activation('Linear'))
    #model.fc0.conv2.register_forward_hook(get_activation('fc0.conv2'))
    #model.fc1.conv2.register_forward_hook(get_activation('fc1.conv2'))


    G = models.Generator(1, nz, 64, 3)  # ngpu, nz, ngf, nc
    D = models.Discriminator(1, 3, 64)  # ngpu, nc, ndf
    global BCE_loss
    global criterion

    BCE_loss = nn.BCELoss()
    criterion = nn.BCELoss()
    if args.cuda:
        model.cuda()
        D.cuda()
        G.cuda()
        criterion.cuda()
        BCE_loss.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    G_optimizer = optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    for epoch in range(1, args.epochs + 1):
        returnLoss, returnKL = train(epoch)
        print(activation['Linear'])

        if returnLoss > 2.0:
            losscounter += 1
        else:
            losscounter = 0
        if losscounter > 5:
            print('losscounter indicates these hyperaparameters are broken, breaking out of this set of parameters.',
                  args.beta)
            badBetas.write(str(args.beta) + '\n')
            badBetas.flush()
            break  # trying to avoid wasting too many epochs on a broken set of hyperparameters
        test(epoch)
        if epoch in decreasing_lr:
            G_optimizer.param_groups[0]['lr'] *= args.droprate
            D_optimizer.param_groups[0]['lr'] *= args.droprate
            optimizer.param_groups[0]['lr'] *= args.droprate
        if epoch % 5 == 0:
            # do checkpointing

            torch.save(G.state_dict(),
                       '%s/%s-%s-%s-%s-_netG%03d.pth' % (
                           args.outf, "cdcgan", args.dataset, args.beta, args.beta, epoch))
            torch.save(D.state_dict(),
                       '%s/%s-%s-%s-%s-_netD%03d.pth' % (
                           args.outf, "cdcgan", args.dataset, args.beta, args.beta, epoch))

            modelName = '%s/%s-%s-%s-%s-model_%03d.pth' % (
                args.outf, "cdcgan", args.dataset, args.beta, args.beta, epoch)
            torch.save(model.state_dict(), modelName)
            print('saving')
            print('generate log from in-distribution data')
            generate_target(model=model, outfile=args.outf, cuda=args.cuda, test_loader=test_loader,
                            nt_test_loader=nt_test_loader)
            print('generate log  from out-of-distribution data')
            generate_non_target(model=model, outfile=args.outf, cuda=args.cuda, test_loader=test_loader,
                                nt_test_loader=nt_test_loader)
            print('calculate metrics')
            fpr, auroc, error, auprin, auprout = callog.metric(args.outf)

            if fpr > maxDict['fpr'] and fpr < 99:
                maxDict['fpr'] = fpr
                modelName = '%s/%s-%s-%s-%.3f-%s-%s.pth' % (
                    args.outf, "cdcgan", args.dataset, "fpr", fpr, args.beta, args.beta)
                torch.save(model.state_dict(), modelName)
                print(modelName, "saved")
            if auroc > maxDict['auroc'] and auroc < 99:
                maxDict['auroc'] = auroc
                modelName = '%s/%s-%s-%s-%.3f-%s-%s.pth' % (
                    args.outf, "cdcgan", args.dataset, "auroc", auroc, args.beta, args.beta)
                torch.save(model.state_dict(), modelName)
                print(modelName, "saved")

            if error > maxDict['error'] and error < 99:
                maxDict['error'] = error
                modelName = '%s/%s-%s-%s-%.3f-%s-%s.pth' % (
                    args.outf, "cdcgan", args.dataset, "error", error, args.beta, args.beta)
                torch.save(model.state_dict(), modelName)
                print(modelName, "saved")

            if auprin > maxDict['auprin'] and auprin < 99:
                maxDict['auprin'] = auprin
                modelName = '%s/%s-%s-%s-%.3f-%s-%s.pth' % (
                    args.outf, "cdcgan", args.dataset, "auprin", auprin, args.beta, args.beta)
                torch.save(model.state_dict(), modelName)
                print(modelName, "saved")

            if auprout > maxDict['auprout'] and auprout < 99:
                maxDict['auprout'] = auprout
                modelName = '%s/%s-%s-%s-%.3f-%s-%s.pth' % (
                    args.outf, "cdcgan", args.dataset, "auprout", auprout, args.beta, args.beta)
                torch.save(model.state_dict(), modelName)
                print(modelName, "saved")
            if fpr < .1 or auprout < .1 or auprin < .1 or auroc < .1 or error < .1:
                print('saving error model for debugging')
                modelName = '%s/%s-%s-%s-%s-%s-%s-%s-%s-%s-%.3f-%.3f.pth' % (
                    args.outf, "cdcgan", args.dataset, "debug", fpr, auroc, error, auprin, auprout, epoch, args.beta,
                    args.beta)
                torch.save(model.state_dict(), modelName)
                print(modelName, "saved")
            if fpr >= 99 or auprout >= 99 or auprin >= 99 or auroc >= 99 or error >= 99:
                print('saving error model for debugging')
                modelName = '%s/%s-%s-%s-%s-%s-%s-%s-%s-%s-%.3f-%.3f.pth' % (
                    args.outf, "cdcgan", args.dataset, "debug", fpr, auroc, error, auprin, auprout, epoch, args.beta,
                    args.beta)
                torch.save(model.state_dict(), modelName)
                print(modelName, "saved")
