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

# ResNet 34 (2015)
# Paper: https://arxiv.org/pdf/1512.03385.pdf

import tensorflow
from tensorflow.keras import Model
import tensorflow.keras.layers as layers

def stem(inputs):
    """ Create the Stem Convolution Group
        inputs : input vector
    """
    # First Convolutional layer, where pooled feature maps will be reduced by 75%
    x = layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same', activation='relu', kernel_initializer="he_normal")(inputs)
    x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    return x
    
def residual_group(x, n_filters, n_blocks, conv=True):
    """ Create a Residual Group
        x        : input to the group
        n_filters: number of filters
        n_blocks : number of blocks in the group
        conv     : flag to include the convolution block connector
    """
    for _ in range(n_blocks):
        x = residual_block(x, n_filters)

    # Double the size of filters and reduce feature maps by 75% (strides=2, 2) to fit the next Residual Group
    if conv:
        x = conv_block(x, n_filters * 2)
    return x

def residual_block(x, n_filters):
    """ Create a Residual Block of Convolutions
        x        : input into the block
        n_filters: number of filters
    """
    shortcut = x
    x = layers.Conv2D(n_filters, (3, 3), strides=(1, 1), padding="same",
                      activation="relu", kernel_initializer="he_normal")(x)
    x = layers.Conv2D(n_filters, (3, 3), strides=(1, 1), padding="same",
                      activation="relu", kernel_initializer="he_normal")(x)
    x = layers.add([shortcut, x])
    return x

def conv_block(x, n_filters):
    """ Create Block of Convolutions without Pooling
        x        : input into the block
        n_filters: number of filters
    """
    x = layers.Conv2D(n_filters, (3, 3), strides=(2, 2), padding="same",
                  activation="relu", kernel_initializer="he_normal")(x)
    x = layers.Conv2D(n_filters, (3, 3), strides=(2, 2), padding="same",
                  activation="relu", kernel_initializer="he_normal")(x)
    return x
    
def classifier(x, n_classes):
    """ Create the Classifier Group
        x         : input vector
        n_classes : number of output classes
    """
    # Pool at the end of all the convolutional residual blocks
    x = layers.GlobalAveragePooling2D()(x)

    # Final Dense Outputting Layer for the outputs
    outputs = layers.Dense(n_classes, activation='softmax')(x)
    return outputs

# The input tensor
inputs = layers.Input(shape=(224, 224, 3))

# The Stem Convolution Group
x = stem(inputs)

# First Residual Block Group of 64 filters
x = residual_group(x, 64, 3)

# Second Residual Block Group of 128 filters
x = residual_group(x, 128, 3)

# Third Residual Block Group of 256 filters
x = residual_group(x, 256, 5)

# Fourth Residual Block Group of 512 filters
x = residual_group(x, 512, 2, False)
    
# The Classifier for 1000 classes
outputs = classifier(x, 1000)

# Instantiate the Model
model = Model(inputs, outputs)

