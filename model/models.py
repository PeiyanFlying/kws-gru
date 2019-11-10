# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Modifications Copyright 2017 Arm Inc. All Rights Reserved.
# Added new model definitions for speech command recognition used in
# the paper: https://arxiv.org/pdf/1711.07128.pdf
#
#

# from model.gru import GRUModel
from model.gru2 import GRUModel
from model.crnn import CRNN
#from model.cnn import CNN
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
"""Model definitions for simple speech recognition.

"""

def prepare_model_settings(label_count, sample_rate, clip_duration_ms,
                           window_size_ms, window_stride_ms,
                           dct_coefficient_count):
  """Calculates common settings needed for all models.

  Args:
    label_count: How many classes are to be recognized.
    sample_rate: Number of audio samples per second.
    clip_duration_ms: Length of each audio clip to be analyzed.
    window_size_ms: Duration of frequency analysis window.
    window_stride_ms: How far to move in time between frequency windows.
    dct_coefficient_count: Number of frequency bins to use for analysis.

  Returns:
    Dictionary containing common settings.
  """
  desired_samples = int(sample_rate * clip_duration_ms / 1000)
  window_size_samples = int(sample_rate * window_size_ms / 1000)
  window_stride_samples = int(sample_rate * window_stride_ms / 1000)
  length_minus_window = (desired_samples - window_size_samples)
  if length_minus_window < 0:
    spectrogram_length = 0
  else:
    spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
  fingerprint_size = dct_coefficient_count * spectrogram_length
  return {
      'desired_samples': desired_samples,
      'window_size_samples': window_size_samples,
      'window_stride_samples': window_stride_samples,
      'spectrogram_length': spectrogram_length,
      'dct_coefficient_count': dct_coefficient_count,
      'fingerprint_size': fingerprint_size,
      'label_count': label_count,
      'sample_rate': sample_rate,
  }

def create_model(model_settings, model_architecture,
                 model_size_info, save_act_value=False, save_act_dir=None):
  """Builds a model of the requested architecture compatible with the settings.

  There are many possible ways of deriving predictions from a spectrogram
  input, so this function provides an abstract interface for creating different
  kinds of models in a black-box way. You need to pass in a TensorFlow node as
  the 'fingerprint' input, and this should output a batch of 1D features that
  describe the audio. Typically this will be derived from a spectrogram that's
  been run through an MFCC, but in theory it can be any feature vector of the
  size specified in model_settings['fingerprint_size'].

  The function will build the graph it needs in the current TensorFlow graph,
  and return the tensorflow output that will contain the 'logits' input to the
  softmax prediction process. If training flag is on, it will also return a
  placeholder node that can be used to control the dropout amount.

  See the implementations below for the possible model architectures that can be
  requested.

  Args:
    fingerprint_input: TensorFlow node that will output audio feature vectors.
    model_settings: Dictionary of information about the model.
    model_architecture: String specifying which kind of model to create.
    is_training: Whether the model is going to be used for training.
    runtime_settings: Dictionary of information about the runtime.

  Returns:
    TensorFlow node outputting logits results, and optionally a dropout
    placeholder.

  Raises:
    Exception: If the architecture type isn't recognized.
  """
  if model_architecture =='gru':
      input_size = model_settings['dct_coefficient_count']  # sequence length 10

      # input_time_size = model_settings['spectrogram_length']  # input_size 25

      layer_dim = model_size_info[0]
      gru_units = model_size_info[1]
      num_classes = model_settings['label_count']

      return GRUModel(input_size, gru_units, layer_dim, num_classes, save_act_value, save_act_dir)

      # return nn.GRU(input_time_size, gru_units, layer_dim, num_classes)

  elif model_architecture =='crnn':
      return CRNN()
  #fyd
  elif model_architecture == 'dnn':
      return DNN(model_settings,model_size_info)
  elif model_architecture == 'lstm':
      return LSTM(model_settings, model_size_info)
  elif model_architecture == 'cnn':
      return CNN(model_settings, model_size_info)
  else:
    raise Exception('model_architecture argument "' + model_architecture +
                    '" not recognized, should be one of "single_fc", "conv",' +
                    ' "low_latency_conv", "low_latency_svdf",'+
                    ' "dnn", "cnn", "basic_lstm", "lstm",'+
                    ' "gru", "crnn" or "ds_cnn"')

class DNN(nn.Module):
    def __init__(self,model_settings,model_size_info):
        super(DNN, self).__init__()
        input_size = model_settings['fingerprint_size']
        label_count = model_settings['label_count']
        layer_dim1 = model_size_info[0]
        layer_dim2 = model_size_info[1]
        layer_dim3 = model_size_info[2]
        self.dnn_net=nn.Sequential(
            nn.Linear(input_size, layer_dim1),
            nn.ReLU(),
            nn.Linear(layer_dim1, layer_dim2),
            nn.ReLU(),
            nn.Linear(layer_dim2, layer_dim3),
            nn.ReLU(),
            nn.Linear(layer_dim3, label_count)
        )
    def forward(self,x):
        x = x.view(x.size(0), -1)
        output = self.dnn_net(x)
        return output

class LSTM(nn.Module):
    def __init__(self, model_settings, model_size_info):
        super(LSTM, self).__init__()
        input_size = model_settings['dct_coefficient_count']
        label_count = model_settings['label_count']
        layer_num = model_size_info[0]
        hidden_dim = model_size_info[1]
        self.LSTM = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=layer_num,
            batch_first=True
        )
        self.out = nn.Linear(in_features=hidden_dim, out_features=label_count)

    def forward(self, x):
        # 一下关于shape的注释只针对单项
        # output: [batch_size, time_step, hidden_size]
        # h_n: [num_layers,batch_size, hidden_size] # 虽然LSTM的batch_first为True,但是h_n/c_n的第一维还是num_layers
        # c_n: 同h_n
        output, (h_n, c_n) = self.LSTM(x)
        #print(output.size())
        # output_in_last_timestep=output[:,-1,:] # 也是可以的
        output_in_last_timestep = h_n[-1, :, :]
        # print(output_in_last_timestep.equal(output[:,-1,:])) #ture
        x = self.out(output_in_last_timestep)
        return x

class CNN(nn.Module):
    def __init__(self, model_settings, model_size_info):
        super(CNN, self).__init__()
        layer_dim2 = model_size_info[10]
        layer_dim3 = model_size_info[11]
        label_count = model_settings['label_count']
        input_frequency_size = model_settings['dct_coefficient_count']
        input_time_size = model_settings['spectrogram_length']
        first_filter_count = model_size_info[0]
        first_filter_height = model_size_info[1]  # time axis
        first_filter_width = model_size_info[2]  # frequency axis
        first_filter_stride_y = model_size_info[3]  # time axis
        first_filter_stride_x = model_size_info[4]  # frequency_axis

        second_filter_count = model_size_info[5]
        second_filter_height = model_size_info[6]  # time axis
        second_filter_width = model_size_info[7]  # frequency axis
        second_filter_stride_y = model_size_info[8]  # time axis
        second_filter_stride_x = model_size_info[9]  # frequency_axis

        linear_layer_size = model_size_info[10]
        fc_size = model_size_info[11]
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=model_size_info[0],
                               kernel_size=(model_size_info[1], model_size_info[2]),
                               stride=(model_size_info[3], model_size_info[4]))
        self.bn1 = nn.BatchNorm2d(first_filter_count)
        self.conv2 = nn.Conv2d(in_channels=model_size_info[0], out_channels=model_size_info[5],
                               kernel_size=(model_size_info[6], model_size_info[7]),
                               stride=(model_size_info[8], model_size_info[9]))
        self.bn2 = nn.BatchNorm2d(second_filter_count)
        first_conv_output_width = np.ceil(
            (input_frequency_size - first_filter_width + 1) /
            first_filter_stride_x)
        first_conv_output_height = np.ceil(
            (input_time_size - first_filter_height + 1) /
            first_filter_stride_y)
        second_conv_output_width = np.ceil(
            (first_conv_output_width - second_filter_width + 1) /
            second_filter_stride_x)
        second_conv_output_height = np.ceil(
            (first_conv_output_height - second_filter_height + 1) /
            second_filter_stride_y)
        second_conv_element_count = int(
            second_conv_output_width * second_conv_output_height * second_filter_count)
        self.fc1 = nn.Linear(in_features=second_conv_element_count,out_features=layer_dim2)
        self.fc2 = nn.Linear(in_features=layer_dim2, out_features=layer_dim3)
        self.bn3 = nn.BatchNorm1d(layer_dim3)
        self.fc3 = nn.Linear(in_features=layer_dim3, out_features=label_count)

    def forward(self, x):
        x=x.unsqueeze(1)
        x=self.conv1(x)
        x=self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = x.view(x.size(0),-1)              # reshape tensor
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x
