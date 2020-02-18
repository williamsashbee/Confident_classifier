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


# Training settings
parser = argparse.ArgumentParser(description='Training code - joint confidence')
parser.add_argument('--batch-size', type=int, default=128, help='input batch size for training')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--log-interval', type=int, default=100, help='how many batches to wait before logging training status')
parser.add_argument('--dataset', default='svhn', help='cifar10 | svhn')
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

print('load data: ',args.dataset)
train_loader, test_loader = data_loader.getTargetDataSet(args.dataset, args.batch_size, args.imageSize, args.dataroot)

print('Load model')
model = models.vgg13()
print(model)

print('load GAN')
nz = 100
netCG = models.cGenerator(1, nz, 64, 3) # ngpu, nz, ngf, nc
netCD = models.cDiscriminator(1, 3, 64) # ngpu, nc, ndf
# Initial setup for GAN
real_label = 1
fake_label = 0
criterion = nn.BCELoss()
#fixed_noise = torch.FloatTensor(64, nz, 1, 1).normal_(0, 1)
fixed_noise = torch.randn((128, 100)).view(-1, 100, 1, 1)
if args.cuda:
    model.cuda()
    netCD.cuda()
    netCG.cuda()
    criterion.cuda()
    fixed_noise = fixed_noise.cuda()
fixed_noise = Variable(fixed_noise)

print('Setup optimizer')
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
optimizerD = optim.Adam(netCD.parameters(), lr=args.lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(netCG.parameters(), lr=args.lr, betas=(0.5, 0.999))
decreasing_lr = list(map(int, args.decreasing_lr.split(',')))

onehot = torch.zeros(10, 10)
onehot = onehot.scatter_(1, torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(10,1), 1).view(10, 10, 1, 1)
img_size = 32
fill = torch.zeros([10, 10, img_size, img_size])
for i in range(10):
    fill[i, i, :, :] = 1

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="6"
#os.environ["CUDA_LAUNCH_BLOCKING"]="1"

# Binary Cross Entropy loss
BCE_loss = nn.BCELoss()


def train(epoch):
    model.train()
    for batch_idx, (data, y_labels) in enumerate(train_loader):
        if data.shape[0] != 128:
            print ("data.shape",data.shape)
            break
        gan_target = torch.FloatTensor(y_labels.size()).fill_(0)
        uniform_dist = torch.Tensor(data.size(0), args.num_classes).fill_((1./args.num_classes))

        if args.cuda:
            data, y_labels = data.cuda(), y_labels.type(torch.LongTensor).cuda()
            gan_target, uniform_dist = gan_target.cuda(), uniform_dist.cuda()


        ###########################
        # (1) Update D network    #
        ###########################
        # train with real
        gan_target.fill_(real_label)
        targetv = Variable(gan_target)
        optimizerD.zero_grad()
        #y_ = (torch.rand(data.shape[0], 1) * 10).type(torch.LongTensor).squeeze()

        #y_label_ = onehot[y_]
        #y_label_ =  Variable(y_label_.cuda())
        #ind = [x for x in y_labels]
        y_fill_ = Variable(fill[y_labels.squeeze().tolist()].cuda())
        output = netCD(data, y_fill_)#!!!seems to be working
        #errD_real = criterion(output, targetv)
        #errD_real.backward()
        D_x = output.data.mean()
        y_real_ = torch.ones(128).cuda()
        D_real_loss = BCE_loss(output, y_real_)

        # train with fake
        #noise = torch.FloatTensor(data.size(0), nz, 1, 1).normal_(0, 1).cuda()
        z_ = torch.randn((128, 100)).view(-1, 100, 1, 1)
        y_ = (torch.rand(128, 1) * 10).type(torch.LongTensor).squeeze()
        y_label_ = onehot[y_]
        y_fill_ = fill[y_]
        z_, y_label_, y_fill_ = Variable(z_.cuda()), Variable(y_label_.cuda()), Variable(y_fill_.cuda())
        #if args.cuda:
        #    noise = noise.cuda()
        #noise = Variable(noise)
        #G_result = G(z_, y_label_)
        #D_result = D(G_result, y_fill_).squeeze()

        fake = netCG(z_, y_label_) #maybe random labels?!
        targetv = Variable(gan_target.fill_(fake_label))
        output = netCD(fake.detach(),y_fill_)
        #errD_fake = criterion(output, targetv)
        #errD_fake.backward()
        y_fake_ = torch.zeros(128).cuda()
        D_fake_loss = BCE_loss(output, y_fake_)



        D_G_z1 = output.data.mean()
        #errD = errD_real + errD_fake
        D_train_loss = D_real_loss + D_fake_loss
        D_train_loss.backward()
        optimizerD.step()

        ###########################
        # (2) Update G network    #
        ###########################
        optimizerG.zero_grad()
        # Original GAN loss
        targetv = Variable(gan_target.fill_(real_label))
        output = netCD(fake, y_fill_)#double check this!
        #errG = criterion(output, targetv)
        G_train_loss = BCE_loss(output, y_real_)

        #G_train_loss.backward()

        D_G_z2 = output.data.mean()

        # minimize the true distribution
        #KL_fake_output = F.log_softmax(model(fake))
        #errG_KL = F.kl_div(KL_fake_output, uniform_dist)*args.num_classes
        generator_loss = G_train_loss #+ args.beta*errG_KL
        generator_loss.backward()
        optimizerG.step()

        ###########################
        # (3) Update classifier   #
        ###########################
        # cross entropy loss
        optimizer.zero_grad()
        output = F.log_softmax(model(data))
        loss = F.nll_loss(output.cuda(), y_labels.type(torch.LongTensor).squeeze().cuda())
        # KL divergence
#        noise = torch.FloatTensor(data.size(0), nz, 1, 1).normal_(0, 1).cuda()
#        if args.cuda:
#            noise = noise.cuda()
#        noise = Variable(noise)

        fake = netCG(z_, y_label_)##check this!
        KL_fake_output = F.log_softmax(model(fake))
        KL_loss_fake = F.kl_div(KL_fake_output, uniform_dist)*args.num_classes
        total_loss = loss + args.beta*KL_loss_fake
        total_loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Classification Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, KL fake Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item(), KL_loss_fake.data.item()))
            fake = netCG(fixed_noise,y_label_)
            vutils.save_image(fake.data, '%s/cgan_samples_epoch_%03d.png'%(args.outf, epoch), normalize=True)

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    for data, target in test_loader:
        total += data.size(0)
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        #data, target = Variable(data, volatile=True), Variable(target)
        output = F.log_softmax(model(data))
        target = target.type(torch.LongTensor) #https://discuss.pytorch.org/t/runtimeerror-multi-target-not-supported-newbie/10216/4
        if args.cuda:
            output = output.cuda()
            target = target.cuda()
        output = Variable(output)
        target = Variable(target)
        target = torch.squeeze(target)

        test_loss += F.nll_loss(output, target).data.item()
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(test_loader) # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, total,
        100. * correct / total))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)
    if epoch in decreasing_lr:
        optimizerG.param_groups[0]['lr'] *= args.droprate
        optimizerD.param_groups[0]['lr'] *= args.droprate
        optimizer.param_groups[0]['lr'] *= args.droprate
    if epoch % 20 == 0:
        # do checkpointing
        torch.save(netCG.state_dict(), '%s/netG_epoch_%d.pth' % (args.outf, epoch))
        torch.save(netCD.state_dict(), '%s/netD_epoch_%d.pth' % (args.outf, epoch))
        torch.save(model.state_dict(), '%s/model_epoch_%d.pth' % (args.outf, epoch))
