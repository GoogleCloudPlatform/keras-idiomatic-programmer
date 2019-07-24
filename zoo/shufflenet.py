# Copyright 2019 Google LLC
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ShuffleNet v1.0
# Paper: https://arxiv.org/pdf/1707.01083.pdf

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, Add, BatchNormalization, ReLU
from tensorflow.keras.layers import DepthwiseConv2D, Lambda, Concatenate, AveragePooling2D
import tensorflow.keras.backend as K

def stem(inputs):
  ''' Stem Convolution Group 
      inputs    : input tensor
  '''
  x = Conv2D(24, (3, 3), strides=2, padding='same', use_bias=False)(inputs)
  x = BatchNormalization()(x)
  x = ReLU()(x)
  return x

def stage(x, n_groups, in_channels, out_channels, n_units, stage_id):
  ''' A Shuffle Net stage (group) 
      x           : input tensor
      n_groups    : number of groups to partition channels into
      n_channels  : number of channels (filters)
      n_units:    : number of shuffle units in this stage (block)
  '''

  if stage_id == 2:
    x = Conv2D( int(out_channels * 0.25), (1, 1), strides=1, padding='same', use_bias=False)(x)
  else:
    # first unit uses a strided convolution
    x = shuffle_unit(x, n_groups, in_channels, out_channels, strides=2)

  for _ in range(n_units-1):
    x = shuffle_unit(x, n_groups, out_channels, out_channels, strides=1)
  return x

def shuffle_unit(x, n_groups, in_channels, out_channels, strides=2, reduce_by=0.25):
  ''' A shuffle unit (e.g., shuffle block)
      x            : input tensor
      n_groups     : number of groups to partition the channels into.
      out_channels : the number of output channels (feature maps)
      strides      : number of strides in 3x3 Depthwise Convolution
      reduce_by    : channel reductioni/restore factor at entry 1x1 convolution
  '''
  # remember input for identity/projection shortcut
  shortcut = x

  # Group 1x1 Convolution
  # Note, the reduce_by meta-parameter to reduce the number of output channels
  x = group_conv(x, n_groups, int(reduce_by * out_channels))
  x = ReLU()(x)

  # Do the Shuffle ...
  x = channel_shuffle(x, n_groups)

  # Depthwise 3x3 Convolution
  x = DepthwiseConv2D((3, 3), strides=strides, padding='same', use_bias=False)(x)
  x = BatchNormalization()(x)

  # Group 1x1 Convolution
  # Restore number of out channels (un-reduce)
  x = group_conv(x, n_groups, out_channels)

  # Identify shortcut
  if strides == 1:
    x = Add()([shortcut, x])
  # Projection shortcut
  else:
    shortcut = AveragePooling2D((3, 3), strides=2, padding='same')(shortcut)
    x = Concatenate(axis=1)([shortcut, x])

  return x

def group_conv(x, n_groups, out_channels):
  ''' Channel Group Convolution 
      x            : input tensor
      n_groups     : number of groups to partition input channels into
      out_channels : number of output channels
  '''

  # Calculate the number of input channels
  in_channels = x.shape[3]

  # Derive the number of input channels per group
  grp_in_channels  = in_channels // n_groups
  # Derive the number output channels per group (Note the rounding up)
  grp_out_channels = int(out_channels / n_groups + 0.5)
  
  # Perform convolutional across each channel group
  groups = []
  for i in range(n_groups):
    # Slice the input across channel group
    group = Lambda(lambda x: x[:, :, :, grp_in_channels * i: grp_in_channels * (i + 1)])(x)

    # Perform convolutional on channel group
    conv = Conv2D(grp_out_channels, (1,1), padding='same', strides=1, use_bias=False)(group)
    # Maintain the channel convolutions in a list
    groups.append(conv)

  # Concatenate the outputs of the group channel convolutions together
  x = Concatenate()(groups)
  # Do batch normalization of the concatenated channels (feature maps)
  x = BatchNormalization()(x)
  return x

def channel_shuffle(x, n_groups):
  ''' Implements the Channel Shuffle layer
      x        : input tensor
      n_groups : number of groups that input channels are partitioned into
  '''
  # Get dimensions of the input tensor
  batch, height, width, in_channels = x.shape

  # Derive the number of input channels per group
  grp_in_channels  = in_channels // n_groups

  # Separate out the channel groups
  x = Lambda(lambda z: K.reshape(z, [-1, height, width, n_groups, grp_in_channels]))(x)
  # Transpose the order of the channel groups (i.e., 3, 4 => 4, 3)
  x = Lambda(lambda z: K.permute_dimensions(z, (0, 1, 2, 4, 3)))(x)
  # Restore shape
  x = Lambda(lambda z: K.reshape(z, [-1, height, width, in_channels]))(x)
  return x

def classifier(x, n_classes):
  ''' Classifier group '''
  x = GlobalAveragePooling2D()(x)
  x = Dense(n_classes, activation='softmax')(x)
  return x

# Number of groups to partition channels into
n_groups=2

# input/output channels per number of groups (key)
channels = {
        1: [24, 144, 288, 576],
        2: [24, 200, 400, 800],
        3: [24, 240, 480, 960],
        4: [24, 272, 544, 1088],
        8: [24, 384, 768, 1536]
}[n_groups]

# input tensor
inputs = Input( (224, 224, 3) )

# Create the stem convolution group (referred to as Stage 1)
x = stem(inputs)

# Stage (block) 2
x = stage(x, n_groups, channels[0], channels[1], n_units=4, stage_id=2)

# Stage (block) 3
x = stage(x, n_groups, channels[1], channels[2], n_units=8, stage_id=3)

# Stage (block) 4
x = stage(x, n_groups, channels[2], channels[3], n_units=4, stage_id=4)

outputs = classifier(x)

model = Model(inputs, outputs)
