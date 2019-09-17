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

# ResNet20, 56, 110, 164, 1001 version 1 for CIFAR-10
# Paper: https://arxiv.org/pdf/1512.03385.pdf

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Add, Dense
from tensorflow.keras.layers import AveragePooling2D, Flatten

def stem(inputs):
    ''' Construct the Stem Convolutional Group 
        inputs : the input vector
    '''
    x = Conv2D(16, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x
    
def learner(x, n_blocks):
    """ Construct the Learner
        x          : input to the learner
        n_blocks   : number of blocks in a group
    """
    # First Residual Block Group of 16 filters (Stage 1)
    # Quadruple (4X) the size of filters to fit the next Residual Group
    x = residual_group(x, 16, n_blocks, strides=(1, 1), n=4)

    # Second Residual Block Group of 64 filters (Stage 2)
    # Double (2X) the size of filters and reduce feature maps by 75% (strides=2) to fit the next Residual Group
    x = residual_group(x, 64, n_blocks, n=2)

    # Third Residual Block Group of 64 filters (Stage 3)
    # Double (2X) the size of filters and reduce feature maps by 75% (strides=2) to fit the next Residual Group
    x = residual_group(x, 128, n_blocks, n=2)
    return x

def residual_group(x, n_filters, n_blocks, strides=(2, 2), n=2):
    """ Construct a Residual Group
        x         : input into the group
        n_filters : number of filters for the group
        n_blocks  : number of residual blocks with identity link
        strides   : whether the projection block is a strided convolution
        n         : multiplier for the number of filters out
    """
    # Double the size of filters to fit the first Residual Group
    x = projection_block(x, n_filters, strides=strides, n=n)

    # Identity residual blocks
    for _ in range(n_blocks):
        x = identity_block(x, n_filters, n)
    return x
    
def identity_block(x, n_filters, n=2):
    """ Construct a Bottleneck Residual Block of Convolutions with Identity Shortcut
        x        : input into the block
        n_filters: number of filters
        n        : multiplier for filters out
    """
    # Save input vector (feature maps) for the identity link
    shortcut = x
    
    ## Construct the 1x1, 3x3, 1x1 residual block (fig 3c)

    # Dimensionality reduction
    x = Conv2D(n_filters, (1, 1), strides=(1, 1), kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Bottleneck layer
    x = Conv2D(n_filters, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Dimensionality restoration - increase the number of output filters by 2X or 4X
    x = Conv2D(n_filters * n, (1, 1), strides=(1, 1), kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)

    # Add the identity link (input) to the output of the residual block
    x = Add()([x, shortcut])
    x = ReLU()(x)
    return x

def projection_block(x, n_filters, strides=(2,2), n=2):
    """ Construct Bottleneck Residual Block with Projection Shortcut
        Increase the number of filters by 2X (or 4X on first stage)
        x        : input into the block
        n_filters: number of filters
        strides  : whether the first convolution is strided
        n        : multiplier for number of filters out
    """
    # Construct the projection shortcut
    # Increase filters by 2X (or 4X) to match shape when added to output of block
    shortcut = Conv2D(n_filters * n, (1, 1), strides=strides, kernel_initializer='he_normal')(x)

    ## Construct the 1x1, 3x3, 1x1 convolution block

    # Dimensionality reduction
    x = Conv2D(n_filters, (1, 1), strides=(1,1), kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # Bottleneck layer - feature pooling when strides=(2, 2)
    x = Conv2D(n_filters, (3, 3), strides=strides, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)  
    
    # Dimensionality restoration - increase the number of filters by 2X (or 4X)
    x = Conv2D(n_filters * n, (1, 1), strides=(1, 1), kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)

    # Add the projection shortcut to the output of the residual block
    x = Add()([shortcut, x])
    x = ReLU()(x)
    return x
    
def classifier(x, n_classes):
    ''' Construct the Classifier
        x         : input into the classifier
        n_classes : number of classes
    '''
    # Pool the feature maps after the end of all the residual blocks
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = AveragePooling2D(pool_size=8)(x)
    
    # Flatten into 1D vector
    x = Flatten()(x)

    # Final Dense Outputting Layer 
    outputs = Dense(n_classes, activation='softmax', kernel_initializer='he_normal')(x)
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
n_blocks = ((depth - 2) // 9) - 1

# The input tensor
inputs = Input(shape=(32, 32, 3))

# The Stem Convolution Group
x = stem(inputs)
   
# The learner
x = learner(x, n_blocks)

# The Classifier for 10 classes
outputs = classifier(x, 10)

# Instantiate the Model
model = Model(inputs, outputs)
