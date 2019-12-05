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
        super().__init__(reg=reg, relu=relu)

        # predefined
        if isinstance(n_layers, int):
            if n_layers not in [16, 19]:
                raise Exception("VGG: Invalid value for n_layers")
            blocks = self.groups[n_layers]
        # user defined
        else:
            blocks = n_layers
            
        # The input vector 
        inputs = Input( input_shape )

        # The stem group
        x = self.stem(inputs, reg=reg)

        # The learner
        x = self.learner(x, blocks=blocks, reg=reg)

        # The classifier
        outputs = self.classifier(x, n_classes, reg=reg)

        # Instantiate the Model
        self._model = Model(inputs, outputs)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, _model):
        self._model = _model
    
    def stem(self, inputs, **metaparameters):
        """ Construct the Stem Convolutional Group
            inputs : the input vector
            reg    : kernel regularizer
        """
        reg = metaparameters['reg']

        x = Conv2D(64, (3, 3), strides=(1, 1), padding="same",
                   kernel_initializer=self.init_weights, kernel_regularizer=reg)(inputs)
        x = Composable.ReLU(x)
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

    @staticmethod
    def group(x, init_weights=None, **metaparameters):
        """ Construct a Convolutional Group
            x        : input to the group
            n_layers : number of convolutional layers
            n_filters: number of filters
        """
        n_filters = metaparameters['n_filters']
        n_layers  = metaparameters['n_layers']
        if 'reg' in metaparameters:
            reg = metaparameters['reg']
        else:
            reg = VGG.reg

        if init_weights is None:
            init_weights = VGG.init_weights

        # Block of convolutional layers
        for n in range(n_layers):
            x = Conv2D(n_filters, (3, 3), strides=(1, 1), padding="same",
                       kernel_initializer=init_weights, kernel_regularizer=reg)(x)
            x = Composable.ReLU(x)
        
        # Max pooling at the end of the block
        x = MaxPooling2D(2, strides=(2, 2))(x)
        return x
    
    def classifier(self, x, n_classes, **metaparameters):
        """ Construct the Classifier
            x         : input to the classifier
            n_classes : number of output classes
            reg       : kernel regularizer
        """
        reg = metaparameters['reg']

        # Save the encoding layer
        self.encoding = x

        # Flatten the feature maps
        x = Flatten()(x)

        # Save the embedding layer
        self.embedding = x
    
        # Two fully connected dense layers
        x = Dense(4096, activation='relu', 
                  kernel_initializer=self.init_weights, kernel_regularizer=reg)(x)
        x = Dense(4096, activation='relu', 
                  kernel_initializer=self.init_weights, kernel_regularizer=reg)(x)

        # Output layer for classification 
        x = Dense(n_classes, 
                  kernel_initializer=self.init_weights, kernel_regularizer=reg)(x)
        # Save the probability distribution before softmax
        self.probabilities = x
        outputs = Activation('softmax')(x)
        # Save the probability distribution after softmax
        self.softmax = outputs
        return outputs

# Example of constructing a VGG 16
# vgg = VGG(16)
