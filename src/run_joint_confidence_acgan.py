##############################################
# This code is based on samples from pytorch #
##############################################
# Writer: Kimin Lee 

from __future__ import print_function
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

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
parser.add_argument('--log-interval', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=130)
parser.add_argument('--ndf', type=int, default=85)
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
parser.add_argument('--beta', type=float, default=.01, help='penalty parameter for KL term')
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
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, download=True, transform=transform),
        batch_size=128, shuffle=True)
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


num_labels = 10
if args.dataset == 'cifar10':
    batchSize = 64
else:
    batchSize = 128
imageSize = 32
input = torch.FloatTensor(batchSize, 3, imageSize, imageSize).cuda()
global noise
noise = torch.FloatTensor(batchSize, nz, 1, 1).cuda()
fixed_noise = torch.FloatTensor(batchSize, nz, 1, 1).normal_(0, 1).cuda()
s_label = torch.FloatTensor(batchSize).cuda()
c_label = torch.LongTensor(batchSize).cuda()

real_label = 1
fake_label = 0


input = Variable(input)
noise = Variable(noise)
fixed_noise = Variable(fixed_noise)




first = True


def train(epoch):
    returnLoss = 100.0
    returnKL = 100.0


    model.train()
    # D_train_loss = 0
    # G_train_loss = 3
    trg = 0
    trd = 0

    global first
    global fixed_noise
    global noise
    for batch_idx, (img, label) in enumerate(train_loader):
        img = img.cuda()
        label = label.cuda()
        label_save = label
        ###########################
        # (1) Update D network
        ###########################
        # train with real
        if img.shape[0] != batchSize:
            print('shape problem')
            break
        if first:
            first = False
            print('fixed label:{}'.format(label))
            fixed_noise_ = np.random.normal(0, 1, (batchSize, nz))
            random_onehot = np.zeros((batchSize, nb_label))
            random_onehot[np.arange(batchSize), label.cpu()] = 1
            fixed_noise_[np.arange(batchSize), :nb_label] = random_onehot[np.arange(batchSize)]
            fixed_noise_ = (torch.from_numpy(fixed_noise_))
            fixed_noise_ = fixed_noise_.resize_(batchSize, nz, 1, 1)
            fixed_noise.data.copy_(fixed_noise_)

            vutils.save_image(img, '%s/%s-%s-%s_realReference.png' % (
                args.outf, "acgan", args.dataset, args.beta), normalize=True)

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

        label = np.random.randint(0, nb_label, batch_size)
        noise_ = np.random.normal(0, 1, (batch_size, nz))
        label_onehot = np.zeros((batch_size, nb_label))
        label_onehot[np.arange(batch_size), label] = 1
        noise_[np.arange(batch_size), :nb_label] = label_onehot[np.arange(batch_size)]

        noise_ = (torch.from_numpy(noise_))
        noise_ = noise_.resize_(batch_size, nz, 1, 1)
        noise.data.copy_(noise_)

        c_label.data.resize_(batch_size).copy_(torch.from_numpy(label))
        noise = Variable(noise)
        fake = netG(noise)
        s_label.data.fill_(fake_label)
        s_output, c_output = netD(fake.detach())
        s_errD_fake = s_criterion(s_output, s_label)
        c_errD_fake = c_criterion(c_output, c_label)
        errD_fake = s_errD_fake + c_errD_fake
        errD_fake.backward()
        D_G_z1 = s_output.data.mean()
        errD_total = errD_fake + errD_real
        optimizerD.step()
        ###########################
        # (2) Update G network
        ###########################

        netG.zero_grad()
        s_label.data.fill_(real_label)  # fake labels are real for generator cost
        s_output, c_output = netD(fake)
        s_errG = s_criterion(s_output, s_label)
        c_errG = c_criterion(c_output, c_label)

        errG = s_errG + c_errG
        errG.backward()
        D_G_z2 = s_output.data.mean()


        #        G_train_loss = BCE_loss(s_output, y_real_)

        # minimize the true distribution
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

        c_label.data.resize_(batch_size).copy_(torch.from_numpy(label))
        noise = noise.cuda()
        noise = Variable(noise)
        fake = netG(noise)
        ####fake code


        KL_fake_output = F.log_softmax(model(fake).squeeze(), dim=1)
        uniform_dist = torch.Tensor(img.shape[0], args.num_classes).fill_((1. / args.num_classes)).cuda()
        errG_KL = F.kl_div(KL_fake_output, uniform_dist)*args.num_classes
        errG_KL = args.beta * errG_KL
        errG_KL.backward()
        #generator_loss = G_train_loss + args.beta*errG_KL # 12.0, .65, 0e-8
        errG_total = errG + errG_KL
        optimizerG.step()
        #G_losses.append(G_train_loss.item())
        ###########################
        # (3) Update classifier   #
        ###########################
        # cross entropy loss
        optimizer.zero_grad()
        x_ = Variable(img)

        output = F.log_softmax(model(x_))
        loss = F.nll_loss(output.cuda(), label_save.squeeze())

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
            returnLoss, returnKL = loss.data.item(), KL_loss_fake.data.item()

            print(
                "Epoch {} , s_errD_real loss {:.6f} c_errD_real loss {:.6f} s_errD_fake loss {:.6f} c_errD_fake loss {:.6f} Generator loss {:.6f} traingenerator {:.6f} traindiscriminator {:.6f}".format(
                    epoch, s_errD_real, c_errD_real,s_errD_fake,c_errD_fake, errG_total, trg, trd))
            #print('Classification Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, KL fake Loss: {:.6f}'.format(
            #    epoch, batch_idx * len(data), len(train_loader.dataset),
            #           100. * batch_idx / len(train_loader), loss.data.item(), KL_loss_fake.data.item()))

            print('Classification Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, KL fake Loss: {:.6f}'.format(
                epoch, batch_idx * len(img), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item(), KL_loss_fake.data.item()))
            fake = netG(fixed_noise.cuda())
            vutils.save_image(fake.data, '%s/%s-%s-%s-%s-_epoch_%03d.png' % (
                args.outf, "acgan", args.dataset, args.beta, args.beta, epoch), normalize=True)

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


maxDict = {'fpr': 0.0, 'auroc': 0.0, 'error': 0.0, 'auprin': 0.0, 'auprout': 0.0}
import random
from test_detection import generate_non_target
from test_detection import generate_target
import calculate_log as callog

badBetas = open('%s/badBetas.txt' % args.outf, 'w')
nt_test_loader = data_loader.getNonTargetDataSet(args.out_dataset, args.batch_size, args.imageSize, args.dataroot)

decreasing_lr = list(map(int, args.decreasing_lr.split(',')))
while True:
    losscounter = 0
    global first
    first = True
    #args.beta = random.uniform(0, 30)
    args.beta = 50.0
    print('betas', args.beta)
    print('Load model')
    global model
    model = models.vgg13()

    netG = models.acnetG(nz, ngf, nc)

    if args.netG != '':
        netG.load_state_dict(torch.load(args.netG))
    print(netG)

    netD = models.acnetD(ndf, nc, nb_label)

    if args.netD != '':
        netD.load_state_dict(torch.load(args.netD))
    print(netD)

    global BCE_loss
    global criterion

    BCE_loss = nn.BCELoss()
    criterion = nn.BCELoss()
    s_criterion = nn.BCELoss()
    c_criterion = nn.CrossEntropyLoss()

    if args.cuda:
        model.cuda()
        netD.cuda()
        netG.cuda()
        criterion.cuda()
        BCE_loss.cuda()
        s_criterion = s_criterion.cuda()
        c_criterion = c_criterion.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(0.5, 0.999))

    for epoch in range(1, args.epochs + 1):
        returnLoss, returnKL = train(epoch)

        if returnLoss > 2.0:
            losscounter += 1
        else:
            losscounter = 0
        if losscounter > 10:
            print('losscounter indicates these hyperaparameters are broken, breaking out of this set of parameters.',
                  args.beta)
            badBetas.write(str(args.beta) + '\n')
            badBetas.flush()
            break  # trying to avoid wasting too many epochs on a broken set of hyperparameters
        test(epoch)
        if epoch in decreasing_lr:
            optimizerG.param_groups[0]['lr'] *= args.droprate
            optimizerD.param_groups[0]['lr'] *= args.droprate
            optimizer.param_groups[0]['lr'] *= args.droprate
        if epoch % 5 == 0:
            # do checkpointing

            torch.save(netG.state_dict(),
                       '%s/%s-%s-%s-%s-_netG%03d.pth' % (
                           args.outf, "acgan", args.dataset, args.beta, args.beta, epoch))
            torch.save(netD.state_dict(),
                       '%s/%s-%s-%s-%s-_netD%03d.pth' % (
                           args.outf, "acgan", args.dataset, args.beta, args.beta, epoch))

            modelName = '%s/%s-%s-%s-%s-model_%03d.pth' % (
                args.outf, "acgan", args.dataset, args.beta, args.beta, epoch)
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
                    args.outf, "acgan", args.dataset, "fpr", fpr, args.beta, args.beta)
                torch.save(model.state_dict(), modelName)
                print(modelName, "saved")
            if auroc > maxDict['auroc'] and auroc < 99:
                maxDict['auroc'] = auroc
                modelName = '%s/%s-%s-%s-%.3f-%s-%s.pth' % (
                    args.outf, "acgan", args.dataset, "auroc", auroc, args.beta, args.beta)
                torch.save(model.state_dict(), modelName)
                print(modelName, "saved")

            if error > maxDict['error'] and error < 99:
                maxDict['error'] = error
                modelName = '%s/%s-%s-%s-%.3f-%s-%s.pth' % (
                    args.outf, "acgan", args.dataset, "error", error, args.beta, args.beta)
                torch.save(model.state_dict(), modelName)
                print(modelName, "saved")

            if auprin > maxDict['auprin'] and auprin < 99:
                maxDict['auprin'] = auprin
                modelName = '%s/%s-%s-%s-%.3f-%s-%s.pth' % (
                    args.outf, "acgan", args.dataset, "auprin", auprin, args.beta, args.beta)
                torch.save(model.state_dict(), modelName)
                print(modelName, "saved")

            if auprout > maxDict['auprout'] and auprout < 99:
                maxDict['auprout'] = auprout
                modelName = '%s/%s-%s-%s-%.3f-%s-%s.pth' % (
                    args.outf, "acgan", args.dataset, "auprout", auprout, args.beta, args.beta)
                torch.save(model.state_dict(), modelName)
                print(modelName, "saved")
            if fpr < .1 or auprout < .1 or auprin < .1 or auroc < .1 or error < .1:
                print('saving error model for debugging')
                modelName = '%s/%s-%s-%s-%s-%s-%s-%s-%s-%s-%.3f-%.3f.pth' % (
                    args.outf, "acgan", args.dataset, "debug", fpr, auroc, error, auprin, auprout, epoch, args.beta,
                    args.beta)
                torch.save(model.state_dict(), modelName)
                print(modelName, "saved")
            if fpr >= 99 or auprout >= 99 or auprin >= 99 or auroc >= 99 or error >= 99:
                print('saving error model for debugging')
                modelName = '%s/%s-%s-%s-%s-%s-%s-%s-%s-%s-%.3f-%.3f.pth' % (
                    args.outf, "acgan", args.dataset, "debug", fpr, auroc, error, auprin, auprout, epoch, args.beta,
                    args.beta)
                torch.save(model.state_dict(), modelName)
                print(modelName, "saved")
