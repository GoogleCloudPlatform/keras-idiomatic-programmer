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

# ResNeXt (50, 101, 152)
# Paper: https://arxiv.org/pdf/1611.05431.pdf

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ReLU, BatchNormalization, Add
from tensorflow.keras.layers import Concatenate, Dense, GlobalAveragePooling2D, Lambda

def stem(inputs):
    """ Construct the Stem Convolution Group
        inputs : input vector
    """
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', kernel_initializer='he_normal', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    return x
    
def learner(x, groups, cardinality=32):
    """ Construct the Learner
        x          : input to the learner
        groups     : list of groups: filters in, filters out, number of blocks
        cardinality: width of group convolution
    """
    # First ResNeXt Group (not-strided)
    filters_in, filters_out, n_blocks = groups.pop(0)
    x = group(x, filters_in, filters_out, n_blocks, strides=(1, 1), cardinality=cardinality)

    # Remaining ResNeXt groups
    for filters_in, filters_out, n_blocks in groups:
    	x = group(x, filters_in, filters_out, n_blocks, cardinality=cardinality)
    return x

def group(x, filters_in, filters_out, n_blocks, cardinality=32, strides=(2, 2)):
    """ Construct a Residual group
        x          : input to the group
        filters_in : number of filters  (channels) at the input convolution
        filters_out: number of filters (channels) at the output convolution
        cardinality: width of group convolution
        strides    : whether its a strided convolution
    """
    # Double the size of filters to fit the first Residual Group
    # Reduce feature maps by 75% (strides=2, 2) to fit the next Residual Group
    x = projection_block(x, filters_in, filters_out, strides=strides, cardinality=cardinality)

    # Remaining blocks
    for _ in range(n_blocks):
        x = identity_block(x, filters_in, filters_out, cardinality=cardinality)	
    return x

def identity_block(x, filters_in, filters_out, cardinality=32):
    """ Construct a ResNeXT block with identity link
        x          : input to block
        filters_in : number of filters  (channels) at the input convolution
        filters_out: number of filters (channels) at the output convolution
        cardinality: width of group convolution
    """
    
    # Remember the input
    shortcut = x

    # Dimensionality Reduction
    x = Conv2D(filters_in, (1, 1), strides=(1, 1),
                      padding='same', kernel_initializer='he_normal', use_bias=False)(shortcut)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Cardinality (Wide) Layer (split-transform)
    filters_card = filters_in // cardinality
    groups = []
    for i in range(cardinality):
        group = Lambda(lambda z: z[:, :, :, i * filters_card:i *
                              filters_card + filters_card])(x)
        groups.append(Conv2D(filters_card, (3, 3), strides=(1, 1),
                                    padding='same', kernel_initializer='he_normal', use_bias=False)(group))

    # Concatenate the outputs of the cardinality layer together (merge)
    x = Concatenate()(groups)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Dimensionality restoration
    x = Conv2D(filters_out, (1, 1), strides=(1, 1),
                      padding='same', kernel_initializer='he_normal', use_bias=False)(x)
    x = BatchNormalization()(x)

    # Identity Link: Add the shortcut (input) to the output of the block
    x = Add()([shortcut, x])
    x = ReLU()(x)
    return x

def projection_block(x, filters_in, filters_out, cardinality=32, strides=(2, 2)):
    """ Construct a ResNeXT block with projection shortcut
        x          : input to the block
        filters_in : number of filters  (channels) at the input convolution
        filters_out: number of filters (channels) at the output convolution
        cardinality: width of group convolution
        strides    : whether entry convolution is strided (i.e., (2, 2) vs (1, 1))
    """
    
    # Construct the projection shortcut
    # Increase filters by 2X to match shape when added to output of block
    shortcut = Conv2D(filters_out, (1, 1), strides=strides,
                                 padding='same', kernel_initializer='he_normal')(x)
    shortcut = BatchNormalization()(shortcut)

    # Dimensionality Reduction
    x = Conv2D(filters_in, (1, 1), strides=(1, 1),
                      padding='same', kernel_initializer='he_normal', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Cardinality (Wide) Layer (split-transform)
    filters_card = filters_in // cardinality
    groups = []
    for i in range(cardinality):
        group = Lambda(lambda z: z[:, :, :, i * filters_card:i *
                              filters_card + filters_card])(x)
        groups.append(Conv2D(filters_card, (3, 3), strides=strides,
                                    padding='same', kernel_initializer='he_normal', use_bias=False)(group))

    # Concatenate the outputs of the cardinality layer together (merge)
    x = Concatenate()(groups)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Dimensionality restoration
    x = Conv2D(filters_out, (1, 1), strides=(1, 1),
                      padding='same', kernel_initializer='he_normal', use_bias=False)(x)
    x = BatchNormalization()(x)

    # Identity Link: Add the shortcut (input) to the output of the block
    x = Add()([shortcut, x])
    x = ReLU()(x)
    return x
    
def classifier(x, n_classes):
    """ Construct the Classifier
        x         : input to the classifier
        n_classes : number of output classes
    """
    # Final Dense Outputting Layer 
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(n_classes, activation='softmax', kernel_initializer='he_normal')(x)
    return outputs

# Meta-parameter: number of filters in, out and number of blocks
groups = { 50 : [ (128, 256, 3), (256, 512, 4), (512, 1024, 6),  (1024, 2048, 3)], # ResNeXt 50
           101: [ (128, 256, 3), (256, 512, 4), (512, 1024, 23), (1024, 2048, 3)], # ResNeXt 101
           152: [ (128, 256, 3), (256, 512, 8), (512, 1024, 36), (1024, 2048, 3)]  # ResNeXt 152
         }
    
# Meta-parameter: width of group convolution
cardinality = 32

# The input tensor
inputs = Input(shape=(224, 224, 3))

# The Stem Group
x = stem(inputs)

# The Learner
x = learner(x, groups[50], cardinality)

# The Classifier for 1000 classes
outputs = classifier(x, 1000)

# Instantiate the Model
model = Model(inputs, outputs)
