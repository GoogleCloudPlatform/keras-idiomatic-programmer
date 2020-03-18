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

# ResNetb (50, 101, 152) version 1.5
# Paper: https://arxiv.org/pdf/1512.03385.pdf
# The strides=2 in the projection_block is moved from the 1x1 convolution to the 
# 3x3 convolution. Gained 0.5% more accuracy on ImageNet

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, BatchNormalization
from tensorflow.keras.layers import ReLU, Dense, GlobalAveragePooling2D, Add


def stem(inputs):
    """ Construct the Stem Convolutional Group 
        inputs : the input vector
    """
    # The 224x224 images are zero padded (black - no signal) to be 230x230 images prior to the first convolution
    x = ZeroPadding2D(padding=(3, 3))(inputs)
    
    # First Convolutional layer uses large (coarse) filter
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='valid', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # Pooled feature maps will be reduced by 75%
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    return x

def learner(x, groups):
    """ Construct the Learner
        x     : input to the learner
        groups: list of groups: number of filters and blocks
    """
    # First Residual Block Group (not strided)
    n_filters, n_blocks = groups.pop(0)
    x = group(x, n_filters, n_blocks, strides=(1, 1))

    # Remaining Residual Block Groups (strided)
    for n_filters, n_blocks in groups:
        x = group(x, n_filters, n_blocks)
    return x

def group(x, n_filters, n_blocks, strides=(2, 2)):
    """ Construct a Residual Group
        x         : input into the group
        n_filters : number of filters for the group
        n_blocks  : number of residual blocks with identity link
        strides   : whether the projection block is a strided convolution
    """
    # Double the size of filters to fit the first Residual Group
    x = projection_block(x, n_filters, strides=strides)

    # Identity residual blocks
    for _ in range(n_blocks):
        x = identity_block(x, n_filters)
    return x

def identity_block(x, n_filters):
    """ Construct a Bottleneck Residual Block with Identity Link
        x        : input into the block
        n_filters: number of filters
    """
    
    # Save input vector (feature maps) for the identity link
    shortcut = x
    
    ## Construct the 1x1, 3x3, 1x1 residual block
    
    # Dimensionality reduction
    x = Conv2D(n_filters, (1, 1), strides=(1, 1), use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Bottleneck layer
    x = Conv2D(n_filters, (3, 3), strides=(1, 1), padding="same", use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Dimensionality restoration - increase the number of output filters by 4X
    x = Conv2D(n_filters * 4, (1, 1), strides=(1, 1), use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)

    # Add the identity link (input) to the output of the residual block
    x = Add()([shortcut, x])
    x = ReLU()(x)
    return x

def projection_block(x, n_filters, strides=(2,2)):
    """ Construct Bottleneck Residual Block of Convolutions with Projection Shortcut
        Increase the number of filters by 4X
        x        : input into the block
        n_filters: number of filters
        strides  : whether the first convolution is strided
    """
    # Construct the projection shortcut
    # Increase filters by 4X to match shape when added to output of block
    shortcut = Conv2D(4 * n_filters, (1, 1), strides=strides, use_bias=False, kernel_initializer='he_normal')(x)
    shortcut = BatchNormalization()(shortcut)

    ## Construct the 1x1, 3x3, 1x1 residual block

    # Dimensionality reduction
    x = Conv2D(n_filters, (1, 1), strides=(1,1), use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Bottleneck layer
    # Feature pooling when strides=(2, 2)
    x = Conv2D(n_filters, (3, 3), strides=strides, padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Dimensionality restoration - increase the number of output filters by 4X
    x = Conv2D(4 * n_filters, (1, 1), strides=(1, 1), use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)

    # Add the projection shortcut to the output of the residual block
    x = Add()([x, shortcut])
    x = ReLU()(x)
    return x

def classifier(x, n_classes):
    """ Construct the Classifier Group 
        x         : input to the classifier
        n_classes : number of output classes
    """
    # Pool at the end of all the convolutional residual blocks
    x = GlobalAveragePooling2D()(x)

    # Final Dense Outputting Layer for the outputs
    outputs = Dense(n_classes, activation='softmax', kernel_initializer='he_normal')(x)
    return outputs

# Meta-parameter: list of groups: number of filters and number of blocks
groups = { 50 : [ (64, 3), (128, 4), (256, 6),  (512, 3) ],           # ResNet50
           101: [ (64, 3), (128, 4), (256, 23), (512, 3) ],           # ResNet101
           152: [ (64, 3), (128, 8), (256, 36), (512, 3) ]            # ResNet152
         }

# The input tensor
inputs = Input(shape=(224, 224, 3))

# The stem convolutional group
x = stem(inputs)

# The learner
x = learner(x, groups[50])

# The classifier for 1000 classes
outputs = classifier(x, 1000)

# Instantiate the Model
model = Model(inputs, outputs)
