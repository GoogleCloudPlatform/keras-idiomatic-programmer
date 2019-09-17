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


# MobileNet 224 (2017)
# Paper: https://arxiv.org/pdf/1704.04861.pdf

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import ZeroPadding2D, Conv2D, BatchNormalization, ReLU
from tensorflow.keras.layers import DepthwiseConv2D, GlobalAveragePooling2D, Reshape, Dropout

def stem(inputs, alpha):
    """ Construct the Stem Group
        inputs : input tensor
        alpha  : width multiplier
    """
    # Convolutional block
    x = ZeroPadding2D(padding=((0, 1), (0, 1)))(inputs)
    x = Conv2D(32 * alpha, (3, 3), strides=(2, 2), padding='valid', use_bias=False, kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = ReLU(6.0)(x)

    # Depthwise Separable Convolution Block
    x = depthwise_block(x, 64, alpha, (1, 1))
    return x
    
def learner(x, alpha):
    """ Construct the Learner
        x      : input to the learner
        alpha  : width multiplier
    """
    # First Depthwise Separable Convolution Group
    x = group(x, 128, 2, alpha)

    # Second Depthwise Separable Convolution Group
    x = group(x, 256, 2, alpha)

    # Third Depthwise Separable Convolution Group
    x = group(x, 512, 6, alpha)

    # Fourth Depthwise Separable Convolution Group
    x = group(x, 1024, 2, alpha)
    return x
    
def group(x, n_filters, n_blocks, alpha):
    """ Construct a Depthwise Separable Convolution Group
        x         : input to the group
        n_filters : number of filters
        n_blocks  : number of blocks in the group
        alpha     : width multiplier
    """   
    # In first block, the depthwise convolution is strided - feature map size reduction
    x = depthwise_block(x, n_filters, alpha, strides=(2, 2))
    
    # Remaining blocks
    for _ in range(n_blocks - 1):
        x = depthwise_block(x, n_filters, alpha, strides=(1, 1))
    return x

def depthwise_block(x, n_filters, alpha, strides):
    """ Construct a Depthwise Separable Convolution block
        x         : input to the block
        n_filters : number of filters
        alpha     : width multiplier
        strides   : strides
    """
    # Apply the width filter to the number of feature maps
    filters = int(n_filters * alpha)

    # Strided convolution to match number of filters
    if strides == (2, 2):
        x = ZeroPadding2D(padding=((0, 1), (0, 1)))(x)
        padding = 'valid'
    else:
        padding = 'same'

    # Depthwise Convolution
    x = DepthwiseConv2D((3, 3), strides, padding=padding, use_bias=False, kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = ReLU(6.0)(x)

    # Pointwise Convolution
    x = Conv2D(filters, (1, 1), strides=(1, 1), padding='same', use_bias=False, kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = ReLU(6.0)(x)
    return x

def classifier(x, alpha, dropout, n_classes):
    """ Construct the classifier group
        x         : input to the classifier
        alpha     : width multiplier
        dropout   : dropout percentage
        n_classes : number of output classes
    """
    # Flatten the feature maps into 1D feature maps (?, N)
    x = GlobalAveragePooling2D()(x)

    # Reshape the feature maps to (?, 1, 1, 1024)
    shape = (1, 1, int(1024 * alpha))
    x = Reshape(shape)(x)
    # Perform dropout for preventing overfitting
    x = Dropout(dropout)(x)

    # Use convolution for classifying (emulates a fully connected layer)
    x = Conv2D(n_classes, (1, 1), padding='same', activation='softmax', kernel_initializer='glorot_uniform')(x)
    # Reshape the resulting output to 1D vector of number of classes
    x = Reshape((n_classes, ))(x)
    return x

# Meta-parameter: width multiplier (0 .. 1) for reducing number of filters.
alpha      = 1   

# Meta-parameter: resolution multiplier (0 .. 1) for reducing input size
pho        = 1

# Meta-parameter: dropout rate
dropout    = 0.5 

inputs = Input(shape=(int(224 * pho), int(224 * pho), 3))

# The Stem Group
x = stem(inputs, alpha)    

# The Learner
x = learner(x, alpha)

# The classifier for 1000 classes
outputs = classifier(x, alpha, dropout, 1000)

# Instantiate the Model
model = Model(inputs, outputs)
