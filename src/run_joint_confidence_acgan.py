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
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--log-interval', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=180)
parser.add_argument('--ndf', type=int, default=80)
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")

parser.add_argument('--dataset', default='mnist', help='cifar10 | svhn')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
parser.add_argument('--droprate', type=float, default=0.1, help='learning rate decay')
parser.add_argument('--decreasing_lr', default='60', help='decreasing strategy')
parser.add_argument('--num_classes', type=int, default=10, help='the # of classes')
parser.add_argument('--beta', type=float, default=1, help='penalty parameter for KL term')
parser.add_argument('--out_dataset', default= "svhn", help='out-of-dist dataset: cifar10 | svhn | imagenet | lsun')

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
if args.dataset=='mnist':
    transform = transforms.Compose([
        transforms.Scale(32),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True, transform=transform),
        batch_size=128, shuffle=True)
    test_loader = None
else:
    train_loader, test_loader = data_loader.getTargetDataSet(args.dataset, args.batch_size, args.imageSize, args.dataroot)



print('Load model')
model = models.vgg13().cuda()
print(model)

print('load GAN')


nz = int(args.nz)
ngf = int(args.ngf)
ndf = int(args.ndf)
if args.dataset == 'mnist':
    #nc = 1
    nc=3
    nb_label = 10
else:
    nc = 3
    nb_label = 10

netG = models.acnetG(nz, ngf, nc)

if args.netG != '':
    netG.load_state_dict(torch.load(args.netG))
print(netG)

netD = models.acnetD(ndf, nc, nb_label)

if args.netD != '':
    netD.load_state_dict(torch.load(args.netD))
print(netD)


print('Setup optimizer')
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

decreasing_lr = list(map(int, args.decreasing_lr.split(',')))

num_labels = 10
if args.dataset == 'cifar10':
    batchSize = 64
else:
    batchSize = 128
imageSize = 32
input = torch.FloatTensor(batchSize, 3, imageSize, imageSize)
global noise
noise = torch.FloatTensor(batchSize, nz, 1, 1)
fixed_noise = torch.FloatTensor(batchSize, nz, 1, 1).normal_(0, 1)
s_label = torch.FloatTensor(batchSize)
c_label = torch.LongTensor(batchSize)

real_label = 1
fake_label = 0


s_criterion = nn.BCELoss()
c_criterion = nn.NLLLoss()

if args.cuda:
    netD.cuda()
    netG.cuda()
    s_criterion.cuda()
    c_criterion.cuda()
    input, s_label = input.cuda(), s_label.cuda()
    c_label = c_label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

input = Variable(input)
s_label = Variable(s_label)
c_label = Variable(c_label)
noise = Variable(noise)
fixed_noise = Variable(fixed_noise)
fixed_noise_ = np.random.normal(0, 1, (batchSize, nz))
random_label = np.random.randint(0, nb_label, batchSize)
print('fixed label:{}'.format(random_label))
random_onehot = np.zeros((batchSize, nb_label))
random_onehot[np.arange(batchSize), random_label] = 1
fixed_noise_[np.arange(batchSize), :nb_label] = random_onehot[np.arange(batchSize)]


fixed_noise_ = (torch.from_numpy(fixed_noise_))
fixed_noise_ = fixed_noise_.resize_(batchSize, nz, 1, 1)
fixed_noise.data.copy_(fixed_noise_)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

BCE_loss = nn.BCELoss()

def train(epoch):
    model.train()
    # D_train_loss = 0
    # G_train_loss = 3
    trg = 0
    trd = 0

    global first
    global fixed_noise
    global noise
    global fixed_label
    global fixed_label_base
    global one_hot_zero
    for batch_idx, (img, label) in enumerate(train_loader):
        img = img.cuda()
        label = label.cuda()
        ###########################
        # (1) Update D network
        ###########################
        # train with real
        if img.shape[0] != batchSize:
            print('shape problem')
            break
        netD.zero_grad()
        batch_size = img.size(0)
        input.data.resize_(img.size()).copy_(img)
        s_label.data.resize_(batch_size).fill_(real_label)
        c_label.data.resize_(batch_size).copy_(label.squeeze())
        s_output, c_output = netD(input)
        s_errD_real = s_criterion(s_output, s_label)
        c_errD_real = c_criterion(c_output, c_label)
        errD_real = s_errD_real + c_errD_real
        errD_real.backward()
        D_x = s_output.data.mean()

        #correct, length = test(c_output, c_label)

        # train with fake
        noise.data.resize_(batch_size, nz, 1, 1)
        noise.data.normal_(0, 1)

        label_fake = np.random.randint(0, nb_label, batch_size)
        noise_ = np.random.normal(0, 1, (batch_size, nz))
        label_onehot = np.zeros((batch_size, nb_label))
        label_onehot[np.arange(batch_size), label_fake] = 1
        noise_[np.arange(batch_size), :nb_label] = label_onehot[np.arange(batch_size)]

        noise_ = (torch.from_numpy(noise_))
        noise_ = noise_.resize_(batch_size, nz, 1, 1)
        noise.data.copy_(noise_)

        c_label.data.resize_(batch_size).copy_(torch.from_numpy(label_fake))

        fake = netG(noise)
        s_label.data.fill_(fake_label)
        s_output, c_output = netD(fake.detach())
        s_errD_fake = s_criterion(s_output, s_label)
        c_errD_fake = c_criterion(c_output, c_label)
        errD_fake = s_errD_fake + c_errD_fake

        errD_fake.backward()
        D_G_z1 = s_output.data.mean()
        errD = s_errD_real + s_errD_fake
        optimizerD.step()
        trd += 1
        ###########################
        # (2) Update G network
        ###########################
        netG.zero_grad()
        s_label.data.fill_(real_label)  # fake labels are real for generator cost
        s_output, c_output = netD(fake)
        s_errG = s_criterion(s_output, s_label)
        c_errG = c_criterion(c_output, c_label)

        errG = s_errG + c_errG
        D_G_z2 = s_output.data.mean()

        y_real_ = torch.ones(img.shape[0]).cuda()
        y_real_ = Variable(y_real_)

        #        G_train_loss = BCE_loss(s_output, y_real_)

        # minimize the true distribution
        ####fake code
        noise.data.resize_(batch_size, nz, 1, 1)
        noise.data.normal_(0, 1)
        label_fake = np.random.randint(0, nb_label, batch_size)
        noise_ = np.random.normal(0, 1, (batch_size, nz))
        label_onehot = np.zeros((batch_size, nb_label))
        label_onehot[np.arange(batch_size), label_fake] = 1
        noise_[np.arange(batch_size), :nb_label] = label_onehot[np.arange(batch_size)]

        noise_ = (torch.from_numpy(noise_))
        noise_ = noise_.resize_(batch_size, nz, 1, 1)
        noise.data.copy_(noise_)

        c_label.data.resize_(batch_size).copy_(torch.from_numpy(label_fake))
        noise = noise.cuda()
        noise = Variable(noise)
        fake = netG(noise)
        ####fake code


        KL_fake_output = F.log_softmax(model(fake).squeeze(), dim=1)
        uniform_dist = torch.Tensor(img.shape[0], args.num_classes).fill_((1. / args.num_classes)).cuda()
        errG_KL = F.kl_div(KL_fake_output, uniform_dist)*args.num_classes
        #generator_loss = G_train_loss + args.beta*errG_KL # 12.0, .65, 0e-8
        generator_loss = errG + args.beta*errG_KL
        generator_loss.backward()
        optimizerG.step()
        #G_losses.append(G_train_loss.item())
        ###########################
        # (3) Update classifier   #
        ###########################
        # cross entropy loss
        optimizer.zero_grad()
        x_ = Variable(img)

        output = F.log_softmax(model(x_))
        loss = F.nll_loss(output.cuda(), label.squeeze())

        # KL divergence

        ####

        ####fake code
        noise.data.resize_(batch_size, nz, 1, 1)
        noise.data.normal_(0, 1)

        label = np.random.randint(0, nb_label, batch_size)
        noise_ = np.random.normal(0, 1, (batch_size, nz))
        label_onehot = np.zeros((batch_size, nb_label))
        label_onehot[np.arange(batch_size), label] = 1
        noise_[np.arange(batch_size), :nb_label] = label_onehot[np.arange(batch_size)]

        noise_ = (torch.from_numpy(noise_))
        noise_ = noise_.resize_(batch_size, nz, 1, 1)
        noise.data.copy_(noise_)
        noise = Variable(noise)
        c_label.data.resize_(batch_size).copy_(torch.from_numpy(label))

        fake = netG(noise)
        ####fake code
        # !!!#D_result = D(G_result, y_fill_).squeeze()

        ####

        KL_fake_output = F.log_softmax(model(fake))
        KL_loss_fake = F.kl_div(KL_fake_output, uniform_dist) * args.num_classes

        total_loss = loss + args.beta * KL_loss_fake
        # total_loss = loss
        total_loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "Epoch {} , Descriminator loss {:.6f} Generator loss {:.6f} traingenerator {:.6f} traindiscriminator {:.6f}".format(
                    epoch, errD, errG, trg, trd))
            #print('Classification Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, KL fake Loss: {:.6f}'.format(
            #    epoch, batch_idx * len(data), len(train_loader.dataset),
            #           100. * batch_idx / len(train_loader), loss.data.item(), KL_loss_fake.data.item()))

            # print('Classification Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, KL fake Loss: {:.6f}'.format(
            #   epoch, batch_idx * len(data), len(train_loader.dataset),
            #   100. * batch_idx / len(train_loader), loss.data.item(), KL_loss_fake.data.item()))
            fake = netG(fixed_noise)
            vutils.save_image(fake.data, '%s/%s_acgan_samples_epoch_%03d.png' % (args.outf, args.dataset, epoch), normalize=True)



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
#    test(epoch)
    if epoch in decreasing_lr:
        optimizerG.param_groups[0]['lr'] *= args.droprate
        optimizerD.param_groups[0]['lr'] *= args.droprate
        optimizer.param_groups[0]['lr'] *= args.droprate
    if epoch % 20 == 0:
        # do checkpointing
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (args.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (args.outf, epoch))
        torch.save(model.state_dict(), '%s/model_epoch_%d.pth' % (args.outf, epoch))
