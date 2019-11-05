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

def encoder(inputs):
    """ Construct the Encoder
        inputs : the input vector
    """
    # First Convolution - feature pooling to 1/2H x 1/2W
    x = Conv2D(64, (3, 3), strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Second Convolution - feature pooling to 1/4H x 1/4W
    x = Conv2D(32, (3, 3), strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Third Convolution - feature pooling to 1/8H x 1/8W
    x = Conv2D(32, (3, 3), strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    return x
    
def decoder(x):
    """ Construct the Decoder
      x         : input to the classifier
    """
    # First Deonvolution - feature unpooling to 1/4H x 1/4W
    x = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # First Deonvolution - feature unpooling to 1/2H x 1/2W
    x = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Third Deonvolution - feature unpooling to H x W x 3
    x = Conv2DTranspose(3, (3, 3), strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    return x

# The input tensor
inputs = Input(shape=(32, 32, 3))

# The encoder
x = encoder(inputs)

# The decoder
outputs = decoder(x)

# Instantiate the Model
model = Model(inputs, outputs)
