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

# SqueezeNet v1.0 +n composable (2016)
# Paper: https://arxiv.org/pdf/1602.07360.pdf

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Concatenate, Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D, Activation
from tensorflow.keras.regularizers import l2

class SqueezeNet(object):
    ''' Construct a SqueezeNet Convolutional Neural Network '''
    # Meta-parameter: number of blocks and filters per group
    groups = [ [ { 'n_filters' : 16 }, { 'n_filters' : 16 }, { 'n_filters' : 32 } ], 
               [ { 'n_filters' : 32 }, { 'n_filters' : 48 }, { 'n_filters' : 48 },  { 'n_filters': 64 } ], 
               [ { 'n_filters' : 64 } ] ]

    # Meta-parameter: dropout rate
    dropout = 0.5

    # Meta-parameter: kernel regularizer
    reg = l2(0.001)

    init_weights = 'glorot_uniform'
    _model = None

    def __init__(self, groups=None, dropout=0.5, reg=l2(0.001), input_shape=(224, 224, 3), n_classes=1000):
        ''' Construct a SqueezeNet Convolutional Neural Network
            dropout    : percent of dropout
            groups     : number of filters per block in groups
            input_shape: input shape to the model
            n_classes  : number of output classes
        '''
        if groups is None:
            groups = SqueezeNet.groups

        # The input shape
        inputs = Input(shape=input_shape)

        # The Stem Group
        x = self.stem(inputs, reg=reg)

        # The Learner
        x = self.learner(x, groups=groups, dropout=dropout, reg=reg)

        # The Classifier
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
        ''' Construct the Stem Group  
            inputs: the input tensor
            reg   : kernel regularizer
        '''
        reg = metaparameters['reg']

        x = Conv2D(96, (7, 7), strides=2, padding='same', activation='relu',
                   kernel_initializer=self.init_weights, kernel_regularizer=reg)(inputs)
        x = MaxPooling2D(3, strides=2)(x)
        return x

    def learner(self, x, **metaparameters):
        ''' Construct the Learner
            x      : input to the learner
            groups : number of blocks/filters per group
            dropout: percent of dropout
        '''
        groups  = metaparameters['groups']
        if 'dropout' in metaparameters:
            dropout = metaparameters['dropout']
        else:
            dropout = SqueezeNet.dropout

        last = groups.pop()

        # Add fire groups, progressively increase number of filters
        for group in groups:
        	x = SqueezeNet.group(x, blocks=group, **metaparameters)

        # Last fire block (module)
        x = SqueezeNet.fire_block(x, **last[0], **metaparameters)

        # Dropout is delayed to end of fire groups
        x = Dropout(dropout)(x)
        return x

    @staticmethod
    def group(x, init_weights=None, **metaparameters):
        ''' Construct a Fire Group
            x     : input to the group
            blocks: list of number of filters per fire block (module)
        '''
        blocks = metaparameters['blocks']

        if init_weights is None:
            init_weights = SqueezeNet.init_weights
            
        # Add the fire blocks (modules) for this group
        for block in blocks:
            x = SqueezeNet.fire_block(x, init_weights=init_weights, **block, **metaparameters)

        # Delayed downsampling
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)
        return x

    @staticmethod
    def fire_block(x, init_weights=None, **metaparameters):
        ''' Construct a Fire Block
            x        : input to the block
            n_filters: number of filters
            reg      : kernel regularizer
        '''
        n_filters = metaparameters['n_filters']
        if 'reg' in metaparameters:
            reg = metaparameters['reg']
        else:
            reg = SqueezeNet.reg

        if init_weights is None:
            init_weights = SqueezeNet.init_weights
            
        # squeeze layer
        squeeze = Conv2D(n_filters, (1, 1), strides=1, activation='relu', padding='same',
                         kernel_initializer=init_weights, kernel_regularizer=reg)(x)

        # branch the squeeze layer into a 1x1 and 3x3 convolution and double the number
        # of filters
        expand1x1 = Conv2D(n_filters * 4, (1, 1), strides=1, activation='relu', padding='same',
                           kernel_initializer=init_weights, kernel_regularizer=reg)(squeeze)
        expand3x3 = Conv2D(n_filters * 4, (3, 3), strides=1, activation='relu', padding='same',
                           kernel_initializer=init_weights, kernel_regularizer=reg)(squeeze)

        # concatenate the feature maps from the 1x1 and 3x3 branches
        x = Concatenate()([expand1x1, expand3x3])
        return x

    def classifier(self, x, n_classes, **metaparameters):
        ''' Construct the Classifier 
            x        : input to the classifier
            n_classes: number of output classes
            reg      : kernel regularizer
        '''
        reg = metaparameters['reg']

        # set the number of filters equal to number of classes
        x = Conv2D(n_classes, (1, 1), strides=1, activation='relu', padding='same',
                   kernel_initializer=self.init_weights, kernel_regularizer=reg)(x)

        # reduce each filter (class) to a single value
        x = GlobalAveragePooling2D()(x)
        x = Activation('softmax')(x)
        return x

# Example
# squeezenet = SqueezeNet()
