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
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling2D, Activation, ReLU, Lambda

def stem(inputs):
    ''' The stem group convolution '''
    # First convolution, use 3x3. Note, no batch normalization is used in stem
    x = Conv2D(24, (3, 3), strides=2, padding='same', activation="relu")(inputs)
    
    # Reduce feature map size by 75%
    x = MaxPooling2D((3, 3), strides=2, padding='same')(x)
    return x
    
def shuffle_block(x, n_groups, in_channels):
    ''' 
        n_groups : number of groups
    '''
    
    # 1x1 Group Convolution
    
    # Calculate the number of filters (channels) per group
    ig = in_channels // n_groups
    
    # Calculate the number of output channels
    out_channels = n_groups * in_channels * 2
    
    # Create a convolution for each group of filters
    groups = []
    for i in range(n_groups):
        offset = i * ig
        group = Lambda(lambda z: z[:, :, :, offset: offset + ig])(x)
        groups.append(Conv2D(int(0.5 + out_channels / n_groups), kernel_size=(1, 1), strides=1)(group))
        
    # Concatenate the filters together
    x = Concatenate()(groups)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    return x
    
def classifier(x):
    ''' '''
    # Now Pool at the end of all the convolutional residual blocks
    x = layers.GlobalAveragePooling2D()(x)

    # Final Dense Outputting Layer for 1000 outputs
    outputs = layers.Dense(1000, activation='softmax')(x)
    
# number of groups in a group convolution
n_groups = 8
    
inputs = Input((224, 224, 3))
x = stem(inputs)

# Get the number of channels from the stem output (should be 24)
input_channels = x.shape[3]
print(x)

# Stage 1
x = shuffle_block(x, n_groups, input_channels)

# Stage 2

# Stage 3
x = shuffle_block(x, 3, 24, 48)
model = Model(inputs, x)
model.summary()