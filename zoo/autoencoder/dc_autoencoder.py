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

# AutoEncoder

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, ReLU, BatchNormalization

def encoder(inputs, layers):
    """ Construct the Encoder
        inputs : the input vector
        layers : number of filters per layer
    """
    x = inputs

    # Feature pooling by 1/2H x 1/2W
    for n_filters in layers:
        x = Conv2D(n_filters, (3, 3), strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

    return x
    
def decoder(x, layers):
    """ Construct the Decoder
      x      : input to decoder
      layers : the number of filters per layer (in encoder)
    """

    # Feature unpooling by 2H x 2W
    for _ in range(len(layers)-1, 0, -1):
        n_filters = layers[_]
        x = Conv2DTranspose(n_filters, (3, 3), strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

    # Last unpooling, restore number of channels
    x = Conv2DTranspose(3, (3, 3), strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    return x

# metaparameter: number of filters per layer in encoder
layers = [64, 32, 32]

# The input tensor
inputs = Input(shape=(32, 32, 3))

# The encoder
x = encoder(inputs, layers)

# The decoder
outputs = decoder(x, layers)

# Instantiate the Model
model = Model(inputs, outputs)
