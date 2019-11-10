import argparse
import os
import logging
import time
import datetime
from time import strftime
import sys
import uuid

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

from model import models
import input_data
from input_data import GSCDataset

import tensorflow as tf
import numpy as np

from adamW import AdamW
import admm

from utils import *

parser = argparse.ArgumentParser(description='PyTorch Training')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--arch', '-a', type=str, default='gru',
                    help='What model architecture to use')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=10, type=int,
                    metavar='N', help='mini-batch size (default: 100)')
parser.add_argument('--lr', '--learning-rate', default=0.0005, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='checkpoints', type=str)
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--optimizer-type', type=str, default='sgd',
                    help="choose optimizer type: [sgd, adam]")
parser.add_argument('--admm-epochs', type=int, default=5, metavar='N',
                    help='number of interval epochs to update admm (default: 5)')
parser.add_argument('--admm-quant', action='store_true', default=False,
                    help='Choose admm quantization training')#modify
parser.add_argument('--quant-type', type=str, default='fixed',
                    help="define sparsity type: [binary, ternary, fixed]")
parser.add_argument('--reg-lambda', default=1e-4, type=float, metavar='M',
                    help='initial rho for all layers')
parser.add_argument('--verbose', action='store_true', default=False,
                    help='whether to report admm convergence condition')
parser.add_argument('--lr_scheduler', type=str, default='default', help='define lr scheduler')
parser.add_argument('--num-bits', type=int, default=4, metavar='N',
                    help="If use one side fixed number bits, please set bit length")
parser.add_argument('--logger', action='store_true', default=False,
                    help='whether to use logger')
parser.add_argument('--quant_val', type=bool, default=False,
                    help="whether to use quantize model")
parser.add_argument('--act_bits', type=int, default=0,
                    help="activation value bits(default: None)")
parser.add_argument('--act_max', type=str, default=None,
                    help="activation value's integer part's max value(default: None)")
parser.add_argument('--save_act_value', type=bool, default=False,
                    help="whether to save activation value to file")
parser.add_argument('--save_act_dir', type=str, default='./act_value/',
                    help="where to save activation value")
parser.add_argument('--coverage', type=float, default=1.0, 
		    help='the percentage of -128~128 covers the whole range of data')

#BSF modulate
parser.add_argument('--save-name', default='', type=str,
                    help='add the name to model for save')

"""
arguments in tensorflow version

"""
parser.add_argument(
    '--data_url',
    type=str,
    default=None,
    # default='http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz',
    help='Location of speech training data archive on the web.')
parser.add_argument(
    '--data_dir',
    type=str,
    default='/home/dongpe/data/speech_commands_v0.02/',
    help="""\
     Where to download the speech training data to.
     """)
parser.add_argument(
    '--background_volume',
    type=float,
    default=0,
    help="""\
  How loud the background noise should be, between 0 and 1.
  """)
parser.add_argument(
    '--background_frequency',
    type=float,
    default=0,
    help="""\
  How many of the training samples have background noise mixed in.
  """)
parser.add_argument(
    '--silence_percentage',
    type=float,
    default=10.0,
    help="""\
  How much of the training data should be silence.
  """)
parser.add_argument(
    '--unknown_percentage',
    type=float,
    default=10.0,
    help="""\
  How much of the training data should be unknown words.
  """)
parser.add_argument(
    '--time_shift_ms',
    type=float,
    default=100.0,
    help="""\
  Range to randomly shift the training audio by in time.
  """)
parser.add_argument(
    '--testing_percentage',
    type=int,
    default=10,
    help='What percentage of wavs to use as a test set.')
parser.add_argument(
    '--validation_percentage',
    type=int,
    default=10,
    help='What percentage of wavs to use as a validation set.')
parser.add_argument(
    '--sample_rate',
    type=int,
    default=16000,
    help='Expected sample rate of the wavs', )
parser.add_argument(
    '--clip_duration_ms',
    type=int,
    default=1000,
    help='Expected duration in milliseconds of the wavs', )
parser.add_argument(
    '--window_size_ms',
    type=float,
    default=30.0,
    help='How long each spectrogram timeslice is', )
parser.add_argument(
    '--window_stride_ms',
    type=float,
    default=10.0,
    help='How long each spectrogram timeslice is', )
parser.add_argument(
    '--dct_coefficient_count',
    type=int,
    default=40,
    help='How many bins to use for the MFCC fingerprint', )

parser.add_argument(
    '--summaries_dir',
    type=str,
    default='/home/liqin/kws-gru/retrain_logs',
    help='Where to save summary logs for TensorBoard.')
parser.add_argument(
    '--wanted_words',
    type=str,
    default='yes,no,up,down,left,right,on,off,stop,go',
    #default='yes',
    help='Words to use (others will be added to an unknown label)', )

parser.add_argument(
    '--model_size_info',
    type=int,
    nargs="+",
    default=[1, 4],
    help='Model dimensions - different for various models')
# set manual seed for testing
# torch.manual_seed(10086)

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()
    args.save_name='loushu'
    if args.logger:
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        logger = logging.getLogger()
        try:
            os.makedirs("logger", exist_ok=True)
        except TypeError:
            raise Exception("Direction not create!")
        logger.addHandler(
            logging.FileHandler(strftime('logger/GSC_%m-%d-%Y-%H:%M_id_') + str(uuid.uuid4()) + '.log', 'a'))
        global print
        print = logger.info

    print("The config arguments showed as below:")
    print(args)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    print("Current network is {}".format(args.arch))

    # Start a new TensorFlow session.
    sess = tf.InteractiveSession()
    # Begin by making sure we have the training data we need. If you already have
    # training data of your own, use `--data_url= ` on the command line to avoid
    # downloading.
    model_settings = models.prepare_model_settings(
        len(input_data.prepare_words_list(args.wanted_words.split(','))),
        args.sample_rate, args.clip_duration_ms, args.window_size_ms,
        args.window_stride_ms, args.dct_coefficient_count)

    print(model_settings)

    audio_processor = input_data.AudioProcessor(
        args.data_url, args.data_dir, args.silence_percentage,
        args.unknown_percentage,
        args.wanted_words.split(','), args.validation_percentage,
        args.testing_percentage, model_settings)
    # fingerprint_size = model_settings['fingerprint_size']
    # label_count = model_settings['label_count']

    # train_set_size = audio_processor.set_size('training')
    # print('set_size=%d', train_set_size)
    # valid_set_size = audio_processor.set_size('validation')
    # print('set_size=%d', valid_set_size)

    time_shift_samples = int((args.time_shift_ms * args.sample_rate) / 1000)

    # train_loader = torch.utils.data.DataLoader(
    #     GSCDataset(args.data_url, args.data_dir, args.silence_percentage, args.unknown_percentage,
    #                args.wanted_words.split(','), args.validation_percentage, args.testing_percentage,
    #                model_settings, sess, args.arch, mode="training", background_frequency=args.background_frequency,
    #                background_volume_range=args.background_frequency, time_shift=time_shift_samples), shuffle=True,
    #     batch_size=args.batch_size, num_workers=args.workers)
    # print("train set size: {}".format(len(train_loader.dataset)))
    val_loader = torch.utils.data.DataLoader(
        GSCDataset(args.data_url, args.data_dir, args.silence_percentage, args.unknown_percentage,
                   args.wanted_words.split(','), args.validation_percentage, args.testing_percentage,
                   model_settings, sess, args.arch, mode="validation"), batch_size=args.batch_size,
        num_workers=args.workers)
    print("validation set size: {}".format(len(val_loader.dataset)))
    test_loader = torch.utils.data.DataLoader(
        GSCDataset(args.data_url, args.data_dir, args.silence_percentage, args.unknown_percentage,
                   args.wanted_words.split(','), args.validation_percentage, args.testing_percentage,
                   model_settings, sess, args.arch, mode="testing"), batch_size=args.batch_size,
        num_workers=args.workers)
    print("test set size: {}".format(len(test_loader.dataset)))

    #model = models.create_model(model_settings, args.arch, args.model_size_info)
    model = models.create_model(model_settings, args.arch, args.model_size_info, args.save_act_value, args.save_act_dir)
    model.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume,map_location='cuda:0')
            try:
                model.load_state_dict(checkpoint)
            except:
                print("Trying load with dict 'state_dict'")
                try:
                    model.load_state_dict(checkpoint['state_dict'])
                except:
                    print("Cann't load model")
                    return

        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return

    cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    if args.optimizer_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=True)
    elif args.optimizer_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                     weight_decay=args.weight_decay)
    elif args.optimizer_type == "adamw":
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer_type == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), args.lr,
                                        weight_decay=args.weight_decay)
    elif args.optimizer_type == "adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), args.lr,
                                        weight_decay=args.weight_decay)
    elif args.optimizer_type == "adadelta":
        optimizer = torch.optim.Adadelta(model.parameters(), args.lr,
                                         weight_decay=args.weight_decay)
    else:
        raise ValueError("The optimizer type is not defined!")

    if args.evaluate:
        # validate(val_loader, model, criterion)
        validate_by_step(args, audio_processor, model, criterion, model_settings, sess)
        #test(test_loader, model, criterion)
        test_by_step(args, audio_processor, model, criterion, model_settings, sess)
        return

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader),
    #                                                        eta_min=4e-08)

    if args.admm_quant:
        name_list = []

        for name, w in model.named_parameters():
            if "weight" or "bias" in name:
                name_list.append(name)

        print("Quantized Layer name list is :")
        print(", ".join(name_list))

        print("Before quantized:")

        validate_by_step(args, audio_processor, model, criterion, model_settings, sess)

        admm.admm_initialization(args, model, device, name_list, print)
        print("After quantized:")
        validate_quant_by_step(args, audio_processor, model, criterion, model_settings,
                               sess, name_list, device)

        for epoch in range(args.start_epoch, args.epochs):

            if args.lr_scheduler == 'default':
                adjust_learning_rate(optimizer, epoch)

            elif args.lr_scheduler == 'cosine':
                pass

            admm_quant_train_by_step(args, audio_processor, model, criterion, optimizer, epoch, model_settings,
                                     time_shift_samples, sess, name_list, device)

            # evaluate on validation set
            print("After Quantized:")
            prec1, quantized_model = validate_quant_by_step(args, audio_processor, model, criterion, model_settings,
                                                            sess, name_list, device)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            if is_best:
                path_name = os.path.join(args.save_dir,
                                         '{arch}_{type}_{num_bits}bits_quantized_GSC_acc_{prec1:.3f}_{add}.pt'.format(
                                             arch=args.arch, 
                                             type=args.quant_type, num_bits=args.num_bits,
                                             prec1=best_prec1,
                                             add=args.save_name))
                new_path_name = os.path.join(args.save_dir,
                                             '{arch}_{type}_{num_bits}bits_quantized_GSC_acc_{prec1:.3f}_{add}.pt'.format(
                                                 arch=args.arch, type=args.quant_type, num_bits=args.num_bits,
                                                 prec1=prec1,
                                                 add=args.save_name))
                if os.path.isfile(path_name):
                    os.remove(path_name)

                best_prec1 = prec1
                save_checkpoint(quantized_model, new_path_name)
                print("Admm training, best top 1 acc {best_prec1:.3f}".format(best_prec1=best_prec1))
                print("Best testing dataset:")
                test_by_step(args, audio_processor, quantized_model, criterion, model_settings, sess)
            else:
                print("Admm training, best top 1 acc {best_prec1:.3f}, current top 1 acc {prec1:.3f}".format(
                    best_prec1=best_prec1, prec1=prec1))


    else:

        for epoch in range(args.start_epoch, args.epochs):

            if args.lr_scheduler == 'default':
                adjust_learning_rate(optimizer, epoch)
            elif args.lr_scheduler == 'cosine':
                pass
                # scheduler.step()

            # train for one epoch
            train_by_step(args, audio_processor, model, criterion, optimizer, epoch, model_settings, time_shift_samples,
                          sess)

            # evaluate on validation set
            # prec1 = validate(val_loader, model, criterion)
            prec1 = validate_by_step(args, audio_processor, model, criterion, model_settings, sess)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            if is_best:
                path_name = os.path.join(args.save_dir,
                                         '{arch}_GSC_acc_{prec1:.3f}_{add}.pt'.format(
                                             arch=args.arch, prec1=best_prec1,add=args.save_name))
                new_path_name = os.path.join(args.save_dir,
                                             '{arch}_GSC_acc_{prec1:.3f}_{add}.pt'.format(
                                                 arch=args.arch, prec1=prec1,add=args.save_name))
                if os.path.isfile(path_name):
                    os.remove(path_name)
                best_prec1 = prec1
                save_checkpoint(model, new_path_name)
                print("Current best validation accuracy {best_prec1:.3f}".format(best_prec1=best_prec1))
            else:
                print("Current validation accuracy {prec1:.3f}, "
                      "best validation accuracy {best_prec1:.3f}".format(prec1=prec1, best_prec1=best_prec1))

        # test(test_loader, model, criterion)
        test_by_step(args, audio_processor, model, criterion, model_settings, sess)


def train_by_step(args, audio_processor, model, criterion, optimizer, epoch, model_settings, time_shift_samples, sess):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()
    train_set_size = audio_processor.set_size('training')
    max_step_epoch = train_set_size // args.batch_size
    input_frequency_size = model_settings['dct_coefficient_count']  # sequence length 10
    input_time_size = model_settings['spectrogram_length']  # input_size 25

    end = time.time()
    for i in range(0, train_set_size, args.batch_size):
        input, target = audio_processor.get_data(
            args.batch_size, 0, model_settings, args.background_frequency,
            args.background_volume, time_shift_samples, 'training', sess)

        # measure data loading time
        data_time.update(time.time() - end)

        target = torch.Tensor(target).cuda()
        _, target = target.max(dim=1)


        input = input.reshape((-1, input_time_size, input_frequency_size))
        input = torch.Tensor(input).cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # output = output.double
        # loss = loss.double()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.data, input.size(0))
        top1.update(prec1, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i // args.batch_size) % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i // args.batch_size, max_step_epoch, batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))


def admm_quant_train_by_step(args, audio_processor, model, criterion, optimizer, epoch, model_settings,
                             time_shift_samples, sess, name_list, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    ce_losses = AverageMeter()
    mixed_losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()
    train_set_size = audio_processor.set_size('training')
    max_step_epoch = train_set_size // args.batch_size
    input_frequency_size = model_settings['dct_coefficient_count']  # sequence length 10
    input_time_size = model_settings['spectrogram_length']  # input_size 25

    end = time.time()
    for i in range(0, train_set_size, args.batch_size):
        input, target = audio_processor.get_data(
            args.batch_size, 0, model_settings, args.background_frequency,
            args.background_volume, time_shift_samples, 'training', sess)

        # measure data loading time
        data_time.update(time.time() - end)

        target = torch.Tensor(target).cuda()
        _, target = target.max(dim=1)
        input = input.reshape((-1, input_time_size, input_frequency_size))
        input = torch.Tensor(input).cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        ce_loss = criterion(output, target_var)
        admm.z_u_update(args, model, device, epoch, i, name_list, print)  # update Z and U variables
        ce_loss, admm_loss, mixed_loss = admm.append_admm_loss(model, ce_loss)  # append admm losss

        # compute gradient
        optimizer.zero_grad()
        mixed_loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        ce_losses.update(ce_loss.data, input.size(0))
        mixed_losses.update(mixed_loss.data, input.size(0))
        top1.update(prec1, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i // args.batch_size) % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Cross Entropy Loss {ce_loss.val:.4f} ({ce_loss.avg:.4f})\t'
                  'Mixed Loss {mixed_loss.val:.4f} ({mixed_loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i // args.batch_size, max_step_epoch, batch_time=batch_time,
                data_time=data_time, ce_loss=ce_losses, mixed_loss=mixed_losses, top1=top1))


def validate_quant_by_step(args, audio_processor, model, criterion, model_settings, sess, name_list, device):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    quantized_model = models.create_model(model_settings, args.arch, args.model_size_info, args.save_act_value, args.save_act_dir)
    quantized_model.alpha = model.alpha
    quantized_model.Q = model.Q
    quantized_model.Z = model.Z
    quantized_model.load_state_dict(model.state_dict())
    quantized_model.cuda()
    admm.apply_quantization(args, quantized_model, name_list, device)

    quantized_model.eval()
    valid_set_size = audio_processor.set_size('validation')
    max_step_epoch = valid_set_size // args.batch_size
    input_frequency_size = model_settings['dct_coefficient_count']  # sequence length 10
    input_time_size = model_settings['spectrogram_length']  # input_size 25

    end = time.time()
    for i in range(0, valid_set_size, args.batch_size):
        input, target = audio_processor.get_data(args.batch_size, i, model_settings, 0.0,
                                                 0.0, 0, 'validation', sess)
        target = torch.Tensor(target).cuda()
        _, target = target.max(dim=1)
        target = target.cuda()
        input = input.reshape((-1, input_time_size, input_frequency_size))
        input = torch.Tensor(input).cuda()
        with torch.no_grad():
            input_var = torch.autograd.Variable(input.float()).cuda()
            target_var = torch.autograd.Variable(target.long())

        # compute output
        output = quantized_model(input_var)
        loss = criterion(output, target_var)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.data, input.size(0))
        top1.update(prec1, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i // args.batch_size) % args.print_freq == 0:
            print('Validation: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i // args.batch_size, max_step_epoch, batch_time=batch_time, loss=losses,
                top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg, quantized_model


def validate_by_step(args, audio_processor, model, criterion, model_settings, sess):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    valid_set_size = audio_processor.set_size('validation')
    max_step_epoch = valid_set_size // args.batch_size
    input_frequency_size = model_settings['dct_coefficient_count']  # sequence length 10
    input_time_size = model_settings['spectrogram_length']  # input_size 25

    end = time.time()
    miss_count = 0
    target_count = 0
    false_count = 0
    file_count = 0
    outputs = torch.tensor([])
    targets = []
    for i in range(0, valid_set_size, args.batch_size):
        input, target = audio_processor.get_data(args.batch_size, i, model_settings, 0.0,
                                                 0.0, 0, 'validation', sess)
        target = torch.Tensor(target).cuda()
        _, target = target.max(dim=1)
        target = target.cuda()
        input = input.reshape((-1, input_time_size, input_frequency_size))
        input = torch.Tensor(input).cuda()
        with torch.no_grad():
            input_var = torch.autograd.Variable(input.float()).cuda()
            target_var = torch.autograd.Variable(target.long())

        # compute output
        #print('input_var.shape=',input_var.shape)
        output = model(input_var)
        #print('output_var.shape=',output.shape)
        loss = criterion(output, target_var)
        
        output = output.float()
        loss = loss.float()

        if outputs.shape[0] == 0:
            outputs = output
        else:
            outputs = torch.cat([outputs, output], 0)
        targets = targets + target.tolist()
        miss_t, target_t, false_t, file_t = evaluate(output.data, target.tolist())
        
        miss_count += miss_t
        target_count += target_t
        false_count += false_t
        file_count += file_t

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.data, input.size(0))
        top1.update(prec1, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        '''
        if i < args.batch_size*5:
            print(output)
            print(target)
            _, pred = output.topk(1, 1, True, True)
            print(pred.t()[0])
            print('miss_count %d', miss_count)
            print('target_count %d', target_count)
            print('false_count %d', false_count)
            print('file_count %d', file_count)
        '''
        if (i // args.batch_size) % args.print_freq == 0:
            #print('Validation: [{0}/{1}]\t'
            #      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
            #    i // args.batch_size, max_step_epoch, batch_time=batch_time, loss=losses,
            #    top1=top1))
            miss_rate = miss_count / target_count if target_count else -1
            false_alarm_rate = false_count / (file_count - target_count) if (file_count - target_count) else -1
            print('Validation: [{0}/{1}]\t'
                  'miss {miss_rate:.2f}%\t'
                  'far {false_alarm_rate:.2f}%\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i // args.batch_size, max_step_epoch, miss_rate=miss_rate*100, false_alarm_rate=false_alarm_rate*100,
                top1=top1))
    ROC(outputs, targets,args.save_dir)
    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg


def test_by_step(args, audio_processor, model, criterion, model_settings, sess):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    test_set_size = audio_processor.set_size('testing')
    max_step_epoch = test_set_size // args.batch_size
    input_frequency_size = model_settings['dct_coefficient_count']  # sequence length 10
    input_time_size = model_settings['spectrogram_length']  # input_size 25

    end = time.time()
    for i in range(0, test_set_size, args.batch_size):
        input, target = audio_processor.get_data(args.batch_size, i, model_settings, 0.0,
                                                 0.0, 0, 'testing', sess)
        target = torch.Tensor(target).cuda()
        _, target = target.max(dim=1)
        target = target.cuda()
        input = input.reshape((-1, input_time_size, input_frequency_size))
        input = torch.Tensor(input).cuda()
        with torch.no_grad():
            input_var = torch.autograd.Variable(input.float()).cuda()
            target_var = torch.autograd.Variable(target.long())

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.data, input.size(0))
        top1.update(prec1, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i // args.batch_size) % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i // args.batch_size, max_step_epoch, batch_time=batch_time, loss=losses,
                top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg


def test(test_loader, model, criterion):
    """
    Run test evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(test_loader):
        _, target = target.max(dim=1)
        target = target.cuda()
        with torch.no_grad():
            input_var = torch.autograd.Variable(input.float()).cuda()
            target_var = torch.autograd.Variable(target.long())

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.data, input.size(0))
        top1.update(prec1, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, len(test_loader), batch_time=batch_time, loss=losses,
                top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg


def save_checkpoint(model, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    state = {}
    state['state_dict'] = model.state_dict()
    if args.admm_quant:
        state['alpha'] = model.alpha

    torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 10 epochs"""
    lr = args.lr * (0.5 ** (epoch // 20))
    print("learning rate ={}".format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()
