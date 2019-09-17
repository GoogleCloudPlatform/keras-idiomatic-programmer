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

# ResNeXt for CIFAR-10 and CIFAR-100
# Paper: https://arxiv.org/pdf/1611.05431.pdf

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Concatenate, Lambda, Add
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

def stem(inputs):
    """ Construct the stem convolutional group
        inputs : the input vector
    """
    # Stem Convolutional layer
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x
    
def learner(x, cardinality=32):
    """ Construct the Learner
        x          : input to the learner
        cardinality: width of group convolution
    """
    # First ResNeXt Group
    for _ in range(3):
        x = resnext_block(x, 64, 128, cardinality=cardinality)

    # Second ResNeXt Group
    for _ in range(3):
        x = resnext_block(x, 128, 256, cardinality=cardinality)

    # Third ResNeXt Group
    for _ in range(3):
        x = resnext_block(x, 256, 512, cardinality=cardinality)
    return x


def resnext_block(shortcut, filters_in, filters_out, cardinality=32):
    """ Construct a ResNeXT block
        shortcut   : previous layer and shortcut for identity link
        filters_in : number of filters  (channels) at the input convolution
        filters_out: number of filters (channels) at the output convolution
        cardinality: width of group convolution
    """
    # Dimensionality reduction
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

    # If first resnext block in a group, use projection shortcut
    if shortcut.shape[-1] != filters_out:
        # use convolutional layer to double the input size to the block so it
        # matches the output size (so we can add them)
        shortcut = Conv2D(filters_out, (1, 1), strides=(1, 1),
                                 padding='same', kernel_initializer='he_normal', use_bias=False)(shortcut)
        shortcut = BatchNormalization()(shortcut)

    # Identity Link: Add the shortcut (input) to the output of the block
    x = Add()([shortcut, x])
    x = ReLU()(x)
    return x
    
def classifier(x, nclasses):
    """ Construct the classifier
    """
    # Final Dense Outputting Layer for the outputs
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(nclasses, activation='softmax', kernel_initializer='he_normal')(x)
    return outputs

# Meta-parameter: width of group convolutional
cardinality=32

# The input tensor
inputs = Input(shape=(32, 32, 3))

# The Stem Group
x = stem(inputs)

# The learner
x = learner(x, cardinality=cardinality)

# The Classifier for the 10 outputs
outputs = classifier(x, 10)

# Instantiate the Model
model = Model(inputs, outputs)
