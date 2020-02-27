##############################################
# This code is based on samples from pytorch #
##############################################
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
import models
from torchvision import datasets, transforms
from torch.autograd import Variable

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

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
parser.add_argument('--dataset', default='mnist', help='mnist | cifar10 | svhn')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
parser.add_argument('--droprate', type=float, default=0.1, help='learning rate decay')
parser.add_argument('--decreasing_lr', default='60', help='decreasing strategy')
parser.add_argument('--num_classes', type=int, default=10, help='the # of classes')
parser.add_argument('--beta', type=float, default=1, help='penalty parameter for KL term')

args = parser.parse_args()

if args.dataset == 'cifar10':
    args.beta = 0.1
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
G = models.cGenerator(1, nz, 64, 3)  # ngpu, nz, ngf, nc
D = models.cDiscriminator(1, 3, 64)  # ngpu, nc, ndf
G.weight_init(mean=0.0, std=0.02)
D.weight_init(mean=0.0, std=0.02)

# Initial setup for GAN
real_label = 1
fake_label = 0
criterion = nn.BCELoss()
nz = 100
#fixed_noise = torch.FloatTensor(64, nz, 1, 1).normal_(0, 1)

# fixed_noise = torch.randn((128, 100)).view(-1, 100, 1, 1)
if args.cuda:
    model.cuda()
    D.cuda()
    G.cuda()
    criterion.cuda()
    #fixed_noise = fixed_noise.cuda()
#fixed_noise = Variable(fixed_noise)

print('Setup optimizer')
lr = 0.0002
batch_size = 128
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

decreasing_lr = list(map(int, args.decreasing_lr.split(',')))

img_size = 32
num_labels = 10
# os.environ["CUDA_LAUNCH_BLOCKING"]="1"

# Binary Cross Entropy loss
BCE_loss = nn.BCELoss()
# fixed_noise = torch.FloatTensor(64, nz, 1, 1).normal_(0, 1)
fixed_noise = torch.randn((64, 100)).view(-1, 100, 1, 1).cuda()

fixed_label = 0
first = True


def train(epoch):
    model.train()
    # D_train_loss = 0
    # G_train_loss = 3
    trg = 0
    trd = 0

    for batch_idx, (data, y_labels) in enumerate(train_loader):
        global first
        global fixed_noise
        global fixed_label

        if first:
            global first
            global fixed_noise
            global fixed_label

            first = False
            fixed_label = y_labels.squeeze()[:64].type(torch.cuda.LongTensor)
            assert fixed_label.shape == (64,)
            print( "saving fixed_label!")
            vutils.save_image(data[:64], '{}/{}realReference{}.png'.format(args.outf,args.dataset, epoch), normalize=True)
        uniform_dist = torch.Tensor(data.size(0), args.num_classes).fill_((1. / args.num_classes)).cuda()
        x_ = data.cuda()
        assert x_[0, :, :, :].shape == (3, 32, 32)

        # train discriminator D
        D.zero_grad()
        y_ = y_labels
        mini_batch = x_.size()[0]

        y_real_ = torch.ones(mini_batch)
        y_fake_ = torch.zeros(mini_batch)
        y_real_, y_fake_ = Variable(y_real_.cuda()), Variable(y_fake_.cuda())


        D_result = D(x_, y_).squeeze()
        D_real_loss = BCE_loss(D_result, y_real_)

        z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1).cuda()
        y_ = (torch.rand(mini_batch, 1) * num_labels).type(torch.cuda.LongTensor).squeeze()

        z_, y_ = Variable(z_.cuda()), Variable(y_.cuda())

        G_result = G(z_, y_.squeeze())
        D_result = D(G_result, y_).squeeze()

        D_fake_loss = BCE_loss(D_result, y_fake_)
        D_fake_score = D_result.data.mean()

        D_train_loss = D_real_loss + D_fake_loss
        trg += 1
        if D_train_loss > .1:
            trd += 1
            D_train_loss.backward()
            D_optimizer.step()

        # D_losses.append(D_train_loss.item())

        # train generator G
        G.zero_grad()

        z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1).cuda()
        y_ = (torch.rand(mini_batch, 1) * num_labels).type(torch.cuda.LongTensor).squeeze()

        z_, y_ = Variable(z_.cuda()), Variable(y_.cuda())


        G_result = G(z_, y_.squeeze())
        D_result = D(G_result, y_).squeeze()

        G_train_loss = BCE_loss(D_result, y_real_)

        # minimize the true distribution
        # KL_fake_output = F.log_softmax(model(G_result))
        # errG_KL = F.kl_div(KL_fake_output, uniform_dist)*args.num_classes
        # generator_loss = G_train_loss + args.beta*errG_KL # 12.0, .65, 0e-8
        # generator_loss.backward()
        G_train_loss.backward()
        G_optimizer.step()
        ###########################
        # (3) Update classifier   #
        ###########################
        # cross entropy loss

        optimizer.zero_grad()
        x_ = Variable(x_)

        output = F.log_softmax(model(x_))
        loss = F.nll_loss(output.cuda(), y_labels.type(torch.cuda.LongTensor).squeeze())

        # KL divergence

        ####
        z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1).cuda()
        y_ = (torch.rand(mini_batch, 1) * num_labels).type(torch.cuda.LongTensor).squeeze()

        z_, y_ = Variable(z_.cuda()), Variable(y_.cuda())


        G_result = G(z_, y_.squeeze())
        # !!!#D_result = D(G_result, y_fill_).squeeze()

        ####
        KL_fake_output = F.log_softmax(model(G_result))
        KL_loss_fake = F.kl_div(KL_fake_output, uniform_dist) * args.num_classes

        total_loss = loss + args.beta * KL_loss_fake
        # total_loss = loss
        total_loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print(
                "Epoch {} , Descriminator loss {:.6f} Generator loss {:.6f} traingenerator {:.6f} traindiscriminator {:.6f}".format(
                    epoch, D_train_loss, G_train_loss, trg, trd))
            print('Classification Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, KL fake Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data.item(), KL_loss_fake.data.item()))

            # print('Classification Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, KL fake Loss: {:.6f}'.format(
            #   epoch, batch_idx * len(data), len(train_loader.dataset),
            #   100. * batch_idx / len(train_loader), loss.data.item(), KL_loss_fake.data.item()))
            fake = G(fixed_noise, fixed_label)
            vutils.save_image(fake.data, '{}/{}CDCgan_samples_epoch_{}.png'.format(args.outf,args.dataset, epoch), normalize=True)


def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    for data, target in test_loader:
        total += data.size(0)
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        # data, target = Variable(data, volatile=True), Variable(target)
        output = F.log_softmax(model(data))
        target = target.type(
            torch.LongTensor)  # https://discuss.pytorch.org/t/runtimeerror-multi-target-not-supported-newbie/10216/4
        if args.cuda:
            output = output.cuda()
            target = target.cuda()
        target = torch.squeeze(target)

        test_loss += F.nll_loss(output, target).data.item()
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(test_loader)  # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, total,
        100. * correct / total))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)
    if epoch in decreasing_lr:
        G_optimizer.param_groups[0]['lr'] *= args.droprate
        D_optimizer.param_groups[0]['lr'] *= args.droprate
        optimizer.param_groups[0]['lr'] *= args.droprate
    if epoch % 20 == 0:
        # do checkpointing
        torch.save(G.state_dict(), '%s/netG_epoch_%d.pth' % (args.outf, epoch))
        torch.save(D.state_dict(), '%s/netD_epoch_%d.pth' % (args.outf, epoch))
        torch.save(model.state_dict(), '%s/model_epoch_%d.pth' % (args.outf, epoch))
