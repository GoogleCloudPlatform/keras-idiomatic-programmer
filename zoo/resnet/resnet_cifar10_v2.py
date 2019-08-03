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

# ResNet20, 56, 110, 164, 1001 version 2 for CIFAR-10
# Paper: https://arxiv.org/pdf/1603.05027.pdf

import tensorflow as tf
from tensorflow.keras import Model
import tensorflow.keras.layers as layers

def stem(inputs):
    ''' Stem Convolutional Group 
        inputs : the input vector
    '''
    x = layers.Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x
    
def bottleneck_block(n_filters, x, n=2):
    """ Create a Bottleneck Residual Block of Convolutions
        n_filters: number of filters
        x        : input into the block
        n        : multiplier for filters out
    """
    shortcut = x
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(n_filters, (1, 1), strides=(1, 1), kernel_initializer='he_normal')(x)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(n_filters, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal')(x)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(n_filters * n, (1, 1), strides=(1, 1), kernel_initializer='he_normal')(x)

    x = layers.add([x, shortcut])
    return x

def projection_block(n_filters, x, strides=(2,2), n=2):
    """ Create Bottleneck Residual Block with Projection Shortcut
        Increase the number of filters by 2X (or 4X on first stage)
        n_filters: number of filters
        x        : input into the block
        strides  : whether the first convolution is strided
        n        : multiplier for number of filters out
    """
    # construct the projection shortcut
    # increase filters by 2X (or 4X) to match shape when added to output of block
    shortcut = layers.Conv2D(n_filters * n, (1, 1), strides=strides, kernel_initializer='he_normal')(x)

    # construct the 1x1, 3x3, 1x1 convolution block
    if x == 2:
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
    x = layers.Conv2D(n_filters, (1, 1), strides=(1,1), kernel_initializer='he_normal')(x)
    
    # feature pooling when strides=(2, 2)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)  
    x = layers.Conv2D(n_filters, (3, 3), strides=strides, padding='same', kernel_initializer='he_normal')(x)
    
    # increase the number of filters by 2X (or 4X)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(n_filters * n, (1, 1), strides=(1, 1), kernel_initializer='he_normal')(x)

    # add the projection shortcut to the output of the convolution block
    x = layers.add([shortcut, x])
    return x
    
def classifier(x, nclasses):
    ''' Classifier
        x        : input into the classifier
        nclasses : number of classes
    '''
    # Pool the feature maps after the end of all the residual blocks
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.AveragePooling2D(pool_size=8)(x)
    
    # Flatten into 1D vector
    x = layers.Flatten()(x)

    # Final Dense Outputting Layer 
    outputs = layers.Dense(nclasses, activation='softmax')(x)
    return outputs

#-------------------
# Model      | n   |
# ResNet20   | 2   |
# ResNet56   | 6   |
# ResNet110  | 12  |
# ResNet164  | 18  |
# ResNet1001 | 111 |
#
n = 18
depth =  n * 9 + 2
nblocks = ((depth - 2) // 9) - 1

# The input tensor
inputs = layers.Input(shape=(32, 32, 3))

# The Stem Convolution Group
x = stem(inputs)
   
# First Residual Block Group of 16 filters (Stage 1)
# Projection Shortcut residual block   
x = projection_block(16, x, strides=(1,1), n=4)

# Identity Residual blocks
for _ in range(nblocks):
    x = bottleneck_block(16, x, n=4)

# Second Residual Block Group of 64 filters (Stage 2)
# Quadruple the size of filters and reduce feature maps by 75% (strides=2, 2) to fit the next Residual Group
x = projection_block(64, x, strides=(2,2), n=2)

# Identity Residual blocks
for _ in range(nblocks):
    x = bottleneck_block(64, x, 2)

# Third Residual Block Group of 64 filters (Stage 3)
# Double the size of filters and reduce feature maps by 75% (strides=2, 2) to fit the next Residual Group
x = projection_block(128, x, strides=(2,2), n=2)

# Identity Residual blocks
for _ in range(nblocks):
    x = bottleneck_block(128, x, 2)

# The Classifier for 10 classes
outputs = classifier(x, 10)

model = Model(inputs, outputs)

