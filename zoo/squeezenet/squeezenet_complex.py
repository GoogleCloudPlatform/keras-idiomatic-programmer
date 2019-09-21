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

# SqueezeNet v1.0 with complex bypass (i.e., transition convolution on identify link) (2016)
# Paper: https://arxiv.org/pdf/1602.07360.pdf

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Concatenate, Add, Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D, Activation

def stem(inputs):
    ''' Construct the Stem Group
        inputs : input tensor
    '''
    x = Conv2D(96, (7, 7), strides=2, padding='same', activation='relu',
               kernel_initializer='glorot_uniform')(inputs)
    x = MaxPooling2D(3, strides=2)(x)
    return x

def learner(x):
    ''' Construct the Learner
	x  : input to the learner
    '''
    # First Fire group, progressively increase number of filters
    x = group(x, [16, 16, 32])

    # Second Fire group
    x = group(x, [32, 48, 48, 64])

    # Last Fire block
    x = fire_block(x, 64)

    # Dropout is delayed to end of fire modules
    x = Dropout(0.5)(x)
    return x

def group(x, filters):
    ''' Construct a Fire Group
	x      : input to the group
	filters: list of number of filters per block in group
    '''
    for n_filters in filters:
        x = fire_block(x, n_filters)

    # Delayed downsampling
    x = MaxPooling2D((3, 3), strides=2)(x)
    return x


def fire_block(x, n_filters):
    ''' Construct a Fire Block  with complex bypass
	x        : input to the block
	n_filters: number of filters in block
    '''
    # remember the input (identity)
    shortcut = x

    # if the number of input filters does not equal the number of output filters, then use
    # a transition convolution to match the number of filters in identify link to output
    if shortcut.shape[3] != 8 * n_filters:
        shortcut = Conv2D(n_filters * 8, (1, 1), strides=1, activation='relu',
                          padding='same', kernel_initializer='glorot_uniform')(shortcut)

    # squeeze layer
    squeeze = Conv2D(n_filters, (1, 1), strides=1, activation='relu',
                     padding='same', kernel_initializer='glorot_uniform')(x)

    # branch the squeeze layer into a 1x1 and 3x3 convolution and double the number
    # of filters
    expand1x1 = Conv2D(n_filters * 4, (1, 1), strides=1, activation='relu',
                      padding='same', kernel_initializer='glorot_uniform')(squeeze)
    expand3x3 = Conv2D(n_filters * 4, (3, 3), strides=1, activation='relu',
                      padding='same', kernel_initializer='glorot_uniform')(squeeze)

    # concatenate the feature maps from the 1x1 and 3x3 branches
    x = Concatenate()([expand1x1, expand3x3])
    
    # if identity link, add (matrix addition) input filters to output filters
    if shortcut is not None:
        x = Add()([x, shortcut])
    return x

def classifier(x, n_classes):
    ''' Construct the Classifier 
	x        : input to the classifier
        n_classes: number of output classes
    '''
    # set the number of filters equal to number of classes
    x = Conv2D(n_classes, (1, 1), strides=1, activation='relu', padding='same',
               kernel_initializer='glorot_uniform')(x)
    # reduce each filter (class) to a single value
    x = GlobalAveragePooling2D()(x)
    x = Activation('softmax')(x)
    return x

# The input shape
inputs = Input((224, 224, 3))

# The Stem Group
x = stem(inputs)

# The Learner
x = learner(x)

# The Classifier
outputs = classifier(x, 1000)

model = Model(inputs, outputs)
