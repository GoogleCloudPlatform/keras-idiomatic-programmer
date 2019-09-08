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

# ResNeXt101
# Paper: https://arxiv.org/pdf/1611.05431.pdf

import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras import Model

def stem(inputs):
    """ Construct the Stem Convolution Group
        inputs : input vector
    """
    x = layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same', kernel_initializer='he_normal', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    return x
    
def learner(x, cardinality=32):
    """ Construct the Learner
        x          : input to the learner
        cardinality: width of group convolution
    """
    # First ResNeXt Group
    x = residual_group(x, 128, 256, 2, strides=(1, 1), cardinality=cardinality)

    # Second ResNeXt Group
    x = residual_group(x, 256, 512, 3, cardinality=cardinality)

    # Third ResNeXt Group
    x = residual_group(x, 512, 1024, 22, cardinality=cardinality)

    # Fourth ResNeXt Group
    x = residual_group(x, 1024, 2048, 2, cardinality=cardinality)
    return x

def residual_group(x, filters_in, filters_out, n_blocks, cardinality=32, strides=(2, 2)):
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
    x = layers.Conv2D(filters_in, kernel_size=(1, 1), strides=(1, 1),
                      padding='same', kernel_initializer='he_normal', use_bias=False)(shortcut)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Cardinality (Wide) Layer (split-transform)
    filters_card = filters_in // cardinality
    groups = []
    for i in range(cardinality):
        group = layers.Lambda(lambda z: z[:, :, :, i * filters_card:i *
                              filters_card + filters_card])(x)
        groups.append(layers.Conv2D(filters_card, kernel_size=(3, 3), strides=(1, 1),
                                    padding='same', kernel_initializer='he_normal', use_bias=False)(group))

    # Concatenate the outputs of the cardinality layer together (merge)
    x = layers.concatenate(groups)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Dimensionality restoration
    x = layers.Conv2D(filters_out, kernel_size=(1, 1), strides=(1, 1),
                      padding='same', kernel_initializer='he_normal', use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    # Identity Link: Add the shortcut (input) to the output of the block
    x = layers.add([shortcut, x])
    x = layers.ReLU()(x)
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
    shortcut = layers.Conv2D(filters_out, kernel_size=(1, 1), strides=strides,
                                 padding='same', kernel_initializer='he_normal', use_bias=False)(x)
    shortcut = layers.BatchNormalization()(shortcut)

    # Dimensionality Reduction
    x = layers.Conv2D(filters_in, kernel_size=(1, 1), strides=(1, 1),
                      padding='same', kernel_initializer='he_normal', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Cardinality (Wide) Layer (split-transform)
    filters_card = filters_in // cardinality
    groups = []
    for i in range(cardinality):
        group = layers.Lambda(lambda z: z[:, :, :, i * filters_card:i *
                              filters_card + filters_card])(x)
        groups.append(layers.Conv2D(filters_card, kernel_size=(3, 3), strides=strides,
                                    padding='same', kernel_initializer='he_normal', use_bias=False)(group))

    # Concatenate the outputs of the cardinality layer together (merge)
    x = layers.concatenate(groups)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Dimensionality restoration
    x = layers.Conv2D(filters_out, kernel_size=(1, 1), strides=(1, 1),
                      padding='same', kernel_initializer='he_normal', use_bias=False)(x)
    x = layers.BatchNormalization()(x)



    # Identity Link: Add the shortcut (input) to the output of the block
    x = layers.add([shortcut, x])
    x = layers.ReLU()(x)
    return x
    
def classifier(x, n_classes):
    """ Construct the Classifier
        x         : input to the classifier
        n_classes : number of output classes
    """
    # Final Dense Outputting Layer 
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(n_classes, activation='softmax')(x)
    return outputs

# Meta-parameter: the width of the group convolution
cardinality=32

# The input tensor
inputs = layers.Input(shape=(224, 224, 3))

# The Stem Group
x = stem(inputs)

# The Learner
x = learner(x, cardinality)

# The Classifier for 1000 classes
outputs = classifier(x, 1000)

# Instantiate the Model
model = Model(inputs, outputs)
