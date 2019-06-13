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

#SqueezeNet v1.0
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Concatenate, Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D, Activation

def stem(inputs):
    ''' The stem group convolution '''
    x = Conv2D(96, (7, 7), strides=2, padding='same', activation='relu',
               kernel_initializer='glorot_uniform')(inputs)
    x = MaxPooling2D(3, strides=2)(x)
    return x

def fire(x, n_filters):
    ''' Create a fire module '''
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
    return x

def classifier(x, n_classes):
    ''' The classifier '''
    # set the number of filters equal to number of classes
    x = Conv2D(n_classes, (1, 1), strides=1, activation='relu', padding='same',
               kernel_initializer='glorot_uniform')(x)
    # reduce each filter (class) to a single value
    x = GlobalAveragePooling2D()(x)
    x = Activation('softmax')(x)
    return x

# The input shape
inputs = Input((224, 224, 3))

# Create the Stem Group
x = stem(inputs)

# Start Fire modules, progressively increase number of filters
x = fire(x, 16)
x = fire(x, 16)
x = fire(x, 32)

# Delayed downsampling
x = MaxPooling2D((3, 3), strides=2)(x)

x = fire(x, 32)
x = fire(x, 48)
x = fire(x, 48)
x = fire(x, 64)

# Delayed downsampling
x = MaxPooling2D((3, 3), strides=2)(x)

# Last fire module
x = fire(x, 64)

# Dropout is delayed to end of fire modules
x = Dropout(0.5)(x)

# Add the classifier
outputs = classifier(x, 1000)

model = Model(inputs, outputs)
