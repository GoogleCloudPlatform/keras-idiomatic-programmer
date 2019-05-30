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

import keras.layers as layers
from keras import Model

def _resnext_block(shortcut, filters_in, filters_out, cardinality=32):
    """ Construct a ResNeXT block
        shortcut   : previous layer and shortcut for identity link
        filters_in : number of filters  (channels) at the input convolution
        filters_out: number of filters (channels) at the output convolution
        cardinality: width of cardinality layer
    """

    # Bottleneck layer
    x = layers.Conv2D(filters_in, kernel_size=(1, 1), strides=(1, 1),
                      padding='same')(shortcut)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Cardinality (Wide) Layer
    filters_card = filters_in // cardinality
    groups = []
    for i in range(cardinality):
        group = layers.Lambda(lambda z: z[:, :, :, i * filters_card:i *
                              filters_card + filters_card])(x)
        groups.append(layers.Conv2D(filters_card, kernel_size=(3, 3),
                                    strides=(1, 1), padding='same')(group))

    # Concatenate the outputs of the cardinality layer together
    x = layers.concatenate(groups)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Bottleneck layer
    x = layers.Conv2D(filters_out, kernel_size=(1, 1), strides=(1, 1),
                      padding='same')(x)
    x = layers.BatchNormalization()(x)

    # special case for first resnext block
    if shortcut.shape[-1] != filters_out:
        # use convolutional layer to double the input size to the block so it
        # matches the output size (so we can add them)
        shortcut = layers.Conv2D(filters_out, kernel_size=(1, 1), strides=(1, 1),
                                 padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    # Identity Link: Add the shortcut (input) to the output of the block
    x = layers.add([shortcut, x])
    x = layers.ReLU()(x)
    return x

# The input tensor
inputs = layers.Input(shape=(224, 224, 3))

# Stem Convolutional layer
x = layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same')(inputs)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

# First ResNeXt Group
for _ in range(3):
    x = _resnext_block(x, 128, 256)

# strided convolution to match the number of output filters on next block and
# reduce by 2
x = layers.Conv2D(512, kernel_size=(1, 1), strides=(2, 2), padding='same')(x)

# Second ResNeXt Group
for _ in range(4):
    x = _resnext_block(x, 256, 512)

# strided convolution to match the number of output filters on next block and
# reduce by 2
x = layers.Conv2D(1024, kernel_size=(1, 1), strides=(2, 2), padding='same')(x)

# Third ResNeXt Group
for _ in range(6):
    x = _resnext_block(x, 512, 1024)

# strided convolution to match the number of output filters on next block and
# reduce by 2
x = layers.Conv2D(2048, kernel_size=(1, 1), strides=(2, 2), padding='same')(x)

# Fourth ResNeXt Group
for _ in range(3):
    x = _resnext_block(x, 1024, 2048)

# Final Dense Outputting Layer for 1000 outputs
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(1000, activation='softmax')(x)

model = Model(inputs, outputs)

