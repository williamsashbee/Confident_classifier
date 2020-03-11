##############################################
# This code is based on samples from pytorch #
##############################################
# Writer: Kimin Lee 
from __future__ import print_function

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

###from test
import calculate_log as callog
import math

# from torch.utils.serialization import load_lua
from torchvision import datasets, transforms
from torch.nn.parameter import Parameter
from numpy.linalg import inv
###from test


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
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
parser.add_argument('--droprate', type=float, default=0.1, help='learning rate decay')
parser.add_argument('--decreasing_lr', default='60', help='decreasing strategy')
parser.add_argument('--num_classes', type=int, default=10, help='the # of classes')
parser.add_argument('--beta1', type=float, default=1.0, help='penalty parameter for KL term')
parser.add_argument('--beta2', type=float, default=1.0, help='penalty parameter for KL term')
####from test
#parser.add_argument('--batch-size', type=int, default=128, help='batch size')
#parser.add_argument('--seed', type=int, default=1, help='random seed')
#parser.add_argument('--dataroot', required=True, help='path to dataset')
#parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
#parser.add_argument('--outf', default='/home/rack/KM/2017_Codes/overconfidence/test/log_entropy',
#                    help='folder to output images and model checkpoints')
parser.add_argument('--out_dataset', default='svhn', help='out-of-dist dataset: cifar10 | svhn | imagenet | lsun')
#parser.add_argument('--num_classes', type=int, default=10, help='number of classes (default: 10)')
parser.add_argument('--pre_trained_net', default='', help="path to pre trained_net")

####
args = parser.parse_args()

if args.dataset == 'cifar10':
    #    args.beta = 0.1
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

model = None

print('load GAN')

# Initial setup for GAN
real_label = 1
fake_label = 0
criterion = None
nz = 100

print('Setup optimizer')
batch_size = 128

decreasing_lr = list(map(int, args.decreasing_lr.split(',')))

img_size = 32
num_labels = 10
BCE_loss = None
# fixed_noise = torch.FloatTensor(64, nz, 1, 1).normal_(0, 1)
fixed_noise = torch.randn((64, 100)).view(-1, 100, 1, 1).cuda()
global fixed_label
fixed_label = 0
global first
first = True



def initTest():


    #if args.cuda:
    #    torch.cuda.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


    print('load non target data: ', args.out_dataset)
    global nt_test_loader
    nt_test_loader = data_loader.getNonTargetDataSet(args.out_dataset, args.batch_size, args.imageSize, args.dataroot)


initTest()


def train(epoch):
    model.train()
    # D_train_loss = 0
    # G_train_loss = 3
    trg = 0
    trd = 0
    returnLoss = 0.0
    returnKL = 0.0
    for batch_idx, (data, y_labels) in enumerate(train_loader):
        global first
        global fixed_noise
        global fixed_label

        if first:
            global fixed_noise
            global fixed_label

            first = False
            fixed_label = y_labels.squeeze()[:64].type(torch.cuda.LongTensor)
            assert fixed_label.shape == (64,)
            print("saving fixed_label!")
            vutils.save_image(data[:64], '%s/%s-%s-%s-%s_realReference.png' % (
            args.outf, "cdcgan", args.dataset, args.beta1, args.beta2), normalize=True)
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
        # D_fake_score = D_result.data.mean()

        D_train_loss = D_real_loss + D_fake_loss
        D_train_loss.backward()
        D_optimizer.step()
        # D_losses.append(D_train_loss.item())

        # train generator G
        G.zero_grad()

        # z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1).cuda()
        # y_ = (torch.rand(mini_batch, 1) * num_labels).type(torch.cuda.LongTensor).squeeze()

        # z_, y_ = Variable(z_.cuda()), Variable(y_.cuda())

        G_result = G(z_, y_.squeeze())
        D_result = D(G_result, y_).squeeze()

        G_train_loss = BCE_loss(D_result, y_real_)

        # minimize the true distribution
        KL_fake_output = F.log_softmax(model(G_result), dim=1)
        errG_KL = F.kl_div(KL_fake_output, uniform_dist) * args.num_classes
        generator_loss = G_train_loss + args.beta1 * errG_KL  # 12.0, .65, 0e-8
        generator_loss.backward()

        G_optimizer.step()

        ###########################
        # (3) Update classifier   #
        ###########################
        # cross entropy loss
        optimizer.zero_grad()
        x_ = Variable(x_)

        output = F.log_softmax(model(x_), dim=1)
        loss = F.nll_loss(output.cuda(), y_labels.type(torch.cuda.LongTensor).squeeze())

        # KL divergence

        ####
        #        z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1).cuda()
        #        y_ = (torch.rand(mini_batch, 1) * num_labels).type(torch.cuda.LongTensor).squeeze()

        #        z_, y_ = Variable(z_.cuda()), Variable(y_.cuda())

        G_result = G(z_, y_.squeeze())
        # !!!#D_result = D(G_result, y_fill_).squeeze()

        ####
        KL_fake_output = F.log_softmax(model(G_result), dim=1)
        KL_loss_fake = F.kl_div(KL_fake_output, uniform_dist) * args.num_classes

        total_loss = loss + args.beta2 * KL_loss_fake
        # total_loss = loss
        total_loss.backward()

        optimizer.step()

        trg += 1
        trd += 1

        returnLoss , returnKL = loss.data.item(), KL_loss_fake.data.item()
        if batch_idx % args.log_interval == 0:
            print(
                "Epoch {} , Descriminator loss {:.6f} Generator loss {:.6f} traingenerator {:.6f} traindiscriminator {:.6f}".format(
                    epoch, D_train_loss, G_train_loss, trg, trd))
            # print('Classification Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, KL fake Loss: {:.6f}'.format(
            #   epoch, batch_idx * len(data), len(train_loader.dataset),
            #          100. * batch_idx / len(train_loader), loss.data.item(), KL_loss_fake.data.item()))

            print('Classification Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, KL fake Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data.item(), KL_loss_fake.data.item()))
            fake = G(fixed_noise, fixed_label)
            vutils.save_image(fake.data, '%s/%s-%s-%s-%s-_epoch_%03d.png' % (
            args.outf, "cdcgan", args.dataset, args.beta1, args.beta2, epoch), normalize=True)
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








def generate_target():
    model.eval()
    correct = 0
    total = 0
    f1 = open('%s/confidence_Base_In.txt' % args.outf, 'w')

    for data, target in test_loader:
        total += data.size(0)
        # vutils.save_image(data, '%s/target_samples.png'%args.outf, normalize=True)
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        batch_output = model(data)

        # compute the accuracy
        pred = batch_output.data.max(1)[1]
        equal_flag = pred.eq(target.data.type(torch.cuda.LongTensor).squeeze())
        correct += equal_flag.sum()
        for i in range(data.size(0)):
            # confidence score: max_y p(y|x)
            output = batch_output[i].view(1, -1)
            soft_out = F.softmax(output)
            soft_out = torch.max(soft_out.data)
            f1.write("{}\n".format(soft_out))

    print('\n Final Accuracy: {}/{} ({:.2f}%)\n'.format(correct, total, 100. * correct / total))


def generate_non_target():
    model.eval()
    total = 0
    f2 = open('%s/confidence_Base_Out.txt' % args.outf, 'w')

    for data, target in nt_test_loader:
        total += data.size(0)
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        batch_output = model(data)
        for i in range(data.size(0)):
            # confidence score: max_y p(y|x)
            output = batch_output[i].view(1, -1)
            soft_out = F.softmax(output)
            soft_out = torch.max(soft_out.data)
            f2.write("{}\n".format(soft_out))

maxDict = {'fpr':0.0,'auroc':0.0,'error':0.0,'auprin':0.0, 'auprout':0.0}
import random
while True:
    losscounter = 0

    args.beta1 = random.uniform(0,30)
    args.beta2 = random.uniform(0,30)
    print ('betas', args.beta1,args.beta2)
    print('Load model')
    global model
    model = models.vgg13()
    print(model)

    G = models.cGenerator(1, nz, 64, 3)  # ngpu, nz, ngf, nc
    D = models.cDiscriminator(1, 3, 64)  # ngpu, nc, ndf
    G.weight_init(mean=0.0, std=0.02)
    D.weight_init(mean=0.0, std=0.02)
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
        if returnLoss>2.0:
            losscounter+=1
        else:
            losscounter = 0
        if losscounter>5:
            print ('losscounter indicates these hyperaparameters are broken, breaking out of this set of parameters.',args.beta1,args.beta2)
            break #trying to avoid wasting too many epochs on a broken set of hyperparameters
        test(epoch)
        if epoch in decreasing_lr:
            G_optimizer.param_groups[0]['lr'] *= args.droprate
            D_optimizer.param_groups[0]['lr'] *= args.droprate
            optimizer.param_groups[0]['lr'] *= args.droprate
        if epoch % 5 == 0:
            # do checkpointing

            torch.save(G.state_dict(),
                       '%s/%s-%s-%s-%s-_netG%03d.pth' % (
                           args.outf, "cdcgan", args.dataset, args.beta1, args.beta2, epoch))
            torch.save(D.state_dict(),
                       '%s/%s-%s-%s-%s-_netD%03d.pth' % (
                           args.outf, "cdcgan", args.dataset, args.beta1, args.beta2, epoch))

            modelName = '%s/%s-%s-%s-%s-model_%03d.pth' % (args.outf, "cdcgan", args.dataset, args.beta1, args.beta2, epoch)
            torch.save(model.state_dict(), modelName)
            print('saving')
            print('generate log from in-distribution data')
            generate_target()
            print('generate log  from out-of-distribution data')
            generate_non_target()
            print('calculate metrics')
            fpr,auroc, error,auprin, auprout = callog.metric(args.outf)
            print('Load model')

            if fpr >maxDict['fpr']:
                maxDict['fpr'] = fpr
                modelName = '%s/%s-%s-%s-%.3f-%s-%s.pth' % (
                args.outf, "cdcgan", args.dataset,"fpr", fpr, args.beta1, args.beta2)
                torch.save(model.state_dict(), modelName)
                print(modelName, "saved")
            if auroc >maxDict['auroc']:
                maxDict['auroc'] = auroc
                modelName = '%s/%s-%s-%s-%.3f-%s-%s.pth' % (
                args.outf, "cdcgan", args.dataset,"auroc", auroc, args.beta1, args.beta2)
                torch.save(model.state_dict(), modelName)
                print(modelName, "saved")

            if error >maxDict['error']:
                maxDict['error'] = error
                modelName = '%s/%s-%s-%s-%.3f-%s-%s.pth' % (
                args.outf, "cdcgan", args.dataset,"error", error, args.beta1, args.beta2)
                torch.save(model.state_dict(), modelName)
                print(modelName, "saved")

            if auprin >maxDict['auprin']:
                maxDict['auprin'] = auprin
                modelName = '%s/%s-%s-%s-%.3f-%s-%s.pth' % (
                args.outf, "cdcgan", args.dataset,"auprin", auprin, args.beta1, args.beta2)
                torch.save(model.state_dict(), modelName)
                print(modelName, "saved")

            if auprout >maxDict['auprout']:
                maxDict['auprout'] = auprout
                modelName = '%s/%s-%s-%s-%.3f-%s-%s.pth' % (
                args.outf, "cdcgan", args.dataset,"auprout", auprout, args.beta1, args.beta2)
                torch.save(model.state_dict(), modelName)
                print(modelName, "saved")





            #global modelTest
            #odelTest = models.vgg13()
            #modelTest.load_state_dict(torch.load(args.pre_trained_net))
            #if args.cuda:
            #    modelTest.cuda()
            #
            #print(modelTest)


