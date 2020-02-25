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

# AutoEncoder - Pooling with Dense Layers/Hidden Units

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, ReLU, BatchNormalization, Reshape, Flatten

def encoder(x, layers):
    ''' Construct the Encoder 
        x     : input to the encoder
         layers: number of nodes per layer
    '''

    # Flatten the input image
    x = Flatten()(x)

    # Progressive Unit Pooling
    for layer in layers:
        n_nodes = layer['n_nodes']
        x = Dense(n_nodes)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

    # The Encoding
    return x

def decoder(x, layers, input_shape):
    ''' Construct the Decoder
        x     : input to the decoder
        layers: nodes per layer
    '''

    # Progressive Unit Unpooling
    for _ in range(len(layers)-1, 0, -1):
        n_nodes = layers[_]['n_nodes']
        x = Dense(n_nodes)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        # Last unpooling and match shape to input
        units = input_shape[0] * input_shape[1] * input_shape[2]
        x = Dense(units, activation='sigmoid')(x)

        # Reshape back into an image
        outputs = Reshape(input_shape)(x)

        # The decoded image
        return outputs

def example():
    ''' Example for constructing/training an AutoEncoder model on MNIST
    '''
    # Example of constructing an AutoEncoder
    # metaparameter: number of filters per layer
    layers = [ {'n_nodes': 256 }, { 'n_nodes': 128 }, { 'n_nodes': 64 } ]
    
    inputs = Input((28, 28, 1))
    _encoder = encoder(inputs, layers)
    outputs = decoder(_encoder, layers, (28, 28, 1))
    ae = Model(inputs, outputs)

    ae.summary()

    from tensorflow.keras.datasets import mnist
    import numpy as np
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train / 255.0).astype(np.float32)
    x_test  = (x_test  / 255.0).astype(np.float32)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test  = np.expand_dims(x_test, axis=-1)

    ae.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    ae.fit(x_train, x_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)
    ae.evaluate(x_test, x_test)


example()
