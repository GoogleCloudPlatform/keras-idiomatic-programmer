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

import sys
sys.path.append('../')
from models_c import Composable

class AutoEncoder(Composable):
    ''' Construct an AutoEncoder '''
    # metaparameter: number of filters per layer
    layers = [ {'n_filters': 64 }, { 'n_filters': 32 }, { 'n_filters': 16 } ]

    # Initial Hyperparameters
    hyperparameters = { 'initializer': 'he_normal',
                        'regularizer': None,
                        'relu_clip'  : None,
                        'bn_epsilon' : None,
                        'use_bias'   : False
                      }

    def __init__(self, layers=None, input_shape=(32, 32, 3),
                 **hyperparameters):
        ''' Construct an AutoEncoder
            input_shape : input shape to the autoencoder
            layers      : the number of filters per layer
            initializer : kernel initializer
            regularizer : kernel regularizer
            relu_clip   : clip value for ReLU
            bn_epsilon  : epsilon for batch norm
            use_bias    : whether to use bias
        '''
        # Configure base (super) class
        Composable.__init__(self, input_shape, None, self.hyperparameters, **hyperparameters)

        if layers is None:
           layers = self.layers

        # remember the layers
        self.layers = layers

        # remember the input shape
        self.input_shape = input_shape

        inputs = Input(input_shape)
        encoder = self.encoder(inputs, layers=layers)
        outputs = self.decoder(encoder, layers=layers)
        self._model = Model(inputs, outputs)

    def encoder(self, x, **metaparameters):
        ''' Construct the Encoder 
            x     : input to the encoder
            layers: number of filters per layer
        '''
        layers = metaparameters['layers']

        # Progressive Feature Pooling
        for layer in layers:
            n_filters = layer['n_filters']
            x = self.Conv2D(x, n_filters, (3, 3), strides=2, padding='same')
            x = self.BatchNormalization(x)
            x = self.ReLU(x)

        # The Encoding
        return x

    def decoder(self, x, init_weights=None, **metaparameters):
        ''' Construct the Decoder
            x     : input to the decoder
            layers: filters per layer
        '''
        layers = metaparameters['layers']

        # Progressive Feature Unpooling
        for _ in range(len(layers)-1, 0, -1):
            n_filters = layers[_]['n_filters']
            x = self.Conv2DTranspose(x, n_filters, (3, 3), strides=2, padding='same')
            x = self.BatchNormalization(x)
            x = self.ReLU(x)

        # Last unpooling and match shape to input
        x = self.Conv2DTranspose(x, 3, (3, 3), strides=2, padding='same')
        x = self.BatchNormalization(x)
        x = self.ReLU(x)

        # The decoded image
        return x

    def compile(self, optimizer='adam'):
        ''' Compile the model using Mean Square Error loss '''
        self._model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

    def extract(self):
        ''' Extract the pretrained encoder
        '''
        # Get the trained weights from the autoencoder
        weights = self._model.get_weights()

        # Extract out the weights for just the encoder  (6 sets per layer)
        encoder_weights = weights[0 : int((6 * len(self.layers)))]
  
        # Construct a copy the encoder
        inputs = Input(self.input_shape)
        outputs = self.encoder(inputs, layers=self.layers)
        encoder = Model(inputs, outputs)

        # Initialize the encoder with the pretrained weights
        encoder.set_weights(encoder_weights)

        return encoder

# Example autoencoder
# autoencoder = AutoEncoder()

# Train the model, and extract pretrained encoder
# e = autoencoder.extract()

def example():
    ''' Example for constructing/training an AutoEncoder  model on CIFAR-10
    '''
    # Example of constructing an AutoEncoder
    ae = AutoEncoder(input_shape=(32, 32, 3))
    ae.model.summary()

    from tensorflow.keras.datasets import cifar10
    import numpy as np
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = (x_train / 255.0).astype(np.float32)
    x_test  = (x_test  / 255.0).astype(np.float32)

    ae.compile()
    ae.model.fit(x_train, x_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)
    ae.model.evaluate(x_test, x_test)


# example()
