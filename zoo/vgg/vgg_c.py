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

# VGG (16 and 19 & Composable) (2014)
# Trainable params: 138,357,544
# Paper: https://arxiv.org/pdf/1409.1556.pdf

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation
from tensorflow.keras.regularizers import l2

import sys
sys.path.append('../')
from models_c import Composable

class VGG(Composable):
    """ VGG (composable)
    """
    # Meta-parameter: list of groups: number of layers and filter size
    groups = { 16 : [ { 'n_layers': 1, 'n_filters': 64 }, 
                      { 'n_layers': 2, 'n_filters': 128 },
                      { 'n_layers': 3, 'n_filters': 256 },
                      { 'n_layers': 3, 'n_filters': 512 },
                      { 'n_layers': 3, 'n_filters': 512 } ],	# VGG16
               19 : [ { 'n_layers': 1, 'n_filters': 64 }, 
                      { 'n_layers': 2, 'n_filters': 128 },
                      { 'n_layers': 4, 'n_filters': 256 },
                      { 'n_layers': 4, 'n_filters': 512 },
                      { 'n_layers': 4, 'n_filters': 512 } ] }	# VGG19

    init_weights = 'glorot_uniform'

    def __init__(self, n_layers, input_shape=(224, 224, 3), n_classes=1000,
                 reg=None, init_weights='glorot_uniform', relu=None):
        """ Construct a VGG model
            n_layers    : number of layers (16 or 19) or metaparameter for blocks
            input_shape : input shape to the model
            n_classes:  : number of output classes
            reg         : kernel regularizer
            init_weights: kernel initializer
            relu        : max value for ReLU
        """
        # Configure the base (super) class
        super().__init__(init_weights=init_weights, reg=reg, relu=relu)

        # predefined
        if isinstance(n_layers, int):
            if n_layers not in [16, 19]:
                raise Exception("VGG: Invalid value for n_layers")
            blocks = list(self.groups[n_layers])
        # user defined
        else:
            blocks = n_layers
            
        # The input vector 
        inputs = Input(input_shape)

        # The stem group
        x = self.stem(inputs)

        # The learner
        x = self.learner(x, blocks=blocks)

        # The classifier
        outputs = self.classifier(x, n_classes)

        # Instantiate the Model
        self._model = Model(inputs, outputs)

    def stem(self, inputs):
        """ Construct the Stem Convolutional Group
            inputs : the input vector
        """
        x = self.Conv2D(inputs, 64, (3, 3), strides=(1, 1), padding="same")
        x = self.ReLU(x)
        return x
    
    def learner(self, x, **metaparameters):
        """ Construct the (Feature) Learner
            x        : input to the learner
            blocks   : list of groups: filter size and number of conv layers
        """ 
        blocks = metaparameters['blocks']

        # The convolutional groups
        for block in blocks:
            x = self.group(x, **block, **metaparameters)
        return x

    def group(self, x, **metaparameters):
        """ Construct a Convolutional Group
            x        : input to the group
            n_layers : number of convolutional layers
            n_filters: number of filters
        """
        n_filters = metaparameters['n_filters']
        n_layers  = metaparameters['n_layers']
        del metaparameters['n_filters']

        # Block of convolutional layers
        for n in range(n_layers):
            x = self.Conv2D(x, n_filters, (3, 3), strides=(1, 1), padding="same",
                            activation=self.ReLU, **metaparameters)
        
        # Max pooling at the end of the block
        x = MaxPooling2D(2, strides=(2, 2))(x)
        return x
    
    def classifier(self, x, n_classes):
        """ Construct the Classifier
            x         : input to the classifier
            n_classes : number of output classes
        """
        # Save the encoding layer
        self.encoding = x

        # Flatten the feature maps
        x = Flatten()(x)

        # Save the embedding layer
        self.embedding = x
    
        # Two fully connected dense layers
        x = self.Dense(x, 4096, activation=self.ReLU)
        x = self.Dense(x, 4096, activation=self.ReLU)

        outputs = super().classifier(x, n_classes, pooling=None)
        return outputs

# Example stock VGG16
# vgg = VGG(16)

def example():
    ''' Example for constructing/training a VGG model
    '''
    # Example of constructing a mini-VGG
    groups = [ { 'n_layers': 1, 'n_filters': 64 },
               { 'n_layers': 2, 'n_filters': 128 },
               { 'n_layers': 2, 'n_filters': 256 } ]
    vgg = VGG(groups, input_shape=(32, 32, 3), n_classes=10)
    vgg.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
    vgg.model.summary()

    # train on CIFAR-10
    from tensorflow.keras.datasets import cifar10
    import numpy as np
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = (x_train / 255.0).astype(np.float32)

    vgg.model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=1)
