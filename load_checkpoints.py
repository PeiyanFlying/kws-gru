import torch
import os
import argparse
import numpy as np
from model import models

import input_data

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Example')
parser.add_argument('--save-dir', type=str, default="./checkpoints", metavar='N',
                    help='Directory to save checkpoints')
parser.add_argument('--save-model', action='store_true', default=True,
                    help='For Saving the current Model')
parser.add_argument('--load-model-name', type=str, default="gru_ternary_quantized_GSC_acc_93.611.pt", metavar='N',
                    help='For loading the model')
parser.add_argument('-a', '--arch', metavar='ARCH', default='gru',
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: gru)')

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
    default='/home/fanyd/Downloads/google_speech_commands_v0.02',
    help="""\
     Where to download the speech training data to.
     """)
parser.add_argument(
    '--background_volume',
    type=float,
    default=0.1,
    help="""\
  How loud the background noise should be, between 0 and 1.
  """)
parser.add_argument(
    '--background_frequency',
    type=float,
    default=0.8,
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
    default=40.0,
    help='How long each spectrogram timeslice is', )
parser.add_argument(
    '--window_stride_ms',
    type=float,
    default=40.0,
    help='How long each spectrogram timeslice is', )
parser.add_argument(
    '--dct_coefficient_count',
    type=int,
    default=10,
    help='How many bins to use for the MFCC fingerprint', )

parser.add_argument(
    '--summaries_dir',
    type=str,
    default='/home/shlin/KWS_HelloEdge/work/retrain_logs',
    help='Where to save summary logs for TensorBoard.')
parser.add_argument(
    '--wanted_words',
    type=str,
    default='yes,no,up,down,left,right,on,off,stop,go',
    help='Words to use (others will be added to an unknown label)', )

parser.add_argument(
    '--model_size_info',
    type=int,
    nargs="+",
    default=[1, 100],
    help='Model dimensions - different for various models')

args = parser.parse_args()

model_settings = models.prepare_model_settings(
    len(input_data.prepare_words_list(args.wanted_words.split(','))),
    args.sample_rate, args.clip_duration_ms, args.window_size_ms,
    args.window_stride_ms, args.dct_coefficient_count)

print(model_settings)

model = models.create_model(model_settings, args.arch, args.model_size_info)
model.cuda()

model_path = os.path.join(args.save_dir, args.load_model_name)
print(model_path)
model.load_state_dict(torch.load(model_path)["state_dict"],strict=False)#i modify here

for name, weight in model.named_parameters():
    print (name)
    unique, counts = np.unique((weight.cpu().detach().numpy()).flatten(), return_counts=True)
    un_list = np.asarray((unique, counts)).T
    print("Unique quantized weights counts:\n", un_list)
    print(len(un_list))
