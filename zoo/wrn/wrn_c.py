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

# Wide Residual Networks (2016)
# Paper: https://arxiv.org/pdf/1605.07146.pdf 

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, Flatten, BatchNormalization, ReLU, Dropout
from tensorflow.keras.layers import Dense, AveragePooling2D, Add, MaxPooling2D
from tensorflow.keras.regularizers import l2

class WRN(object):
    """ Construct a Wide Residual Convolution Network """
    # Meta-parameter: number of filters per group
    groups = [ { 'n_filters': 16 }, { 'n_filters' : 32 }, { 'n_filters' : 64 } ]
    # Meta-parameter: the number of layers
    depth = 16
    # Meta-parameter: the width factor
    k = 8
    # Meta-parameter: dropout
    dropout = 0
    # Meta-parameter: kernel regularizar
    reg = l2(0.001)

    init_weights = 'he_normal'
    _model = None

    def __init__(self, groups=None, depth=16, k=8, dropout=0, input_shape=(32, 32, 3), n_classes=10, reg=None):
        """ Construct a Wids Residual (Convolutional Neural) Network 
            depth      : number of layers
            k          : width factor
            groups     : number of filters per group
            input_shape: input shape
            n_classes  : number of output classes
            reg        : kernel regularization
        """
        if groups is None:
            groups = self.groups

        # The input tensor
        inputs = Input(input_shape)

        # The stem convolutional group
        x = self.stem(inputs, reg=reg)

        # The learner
        x = self.learner(x, groups=groups, depth=depth, k=k, dropout=dropout, reg=reg)

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

        # Convolutional layer 
        x = Conv2D(16, (3, 3), strides=(1, 1), padding='same', use_bias=False, 
                   kernel_initializer=self.init_weights, kernel_regularizer=reg)(inputs)
        x = BatchNormalization()(x)
        x = ReLU()(x)
    
        return x

    def learner(self, x, **metaparameters):
        """ Construct the Learner
            x     : input to the learner
            groups: number of filters per group
            depth : number of convolutional layers
        """
        groups = metaparameters['groups']
        depth  = metaparameters['depth']

        # calculate the number of blocks from the depth   
        n_blocks = (depth - 4) // 6

        # first group, the projection block is not strided
        x = WRN.group(x, n_blocks=n_blocks, strides=(1, 1), **groups.pop(0), **metaparameters)
        
        # remaining groups
        for group in groups:
            x = WRN.group(x, n_blocks=n_blocks, strides=(2, 2), **group, **metaparameters)
        return x
    
    @staticmethod
    def group(x, init_weights=None, **metaparameters):
        """ Construct a Wide Residual Group
            x         : input into the group
            n_blocks  : number of residual blocks with identity link
        """
        n_blocks  = metaparameters['n_blocks']

        # first block is projection to match the number of input filters to output fitlers for the add operation
        x = WRN.projection_block(x, init_weights=init_weights, **metaparameters)

        # wide residual blocks
        for _ in range(n_blocks-1):
            x = WRN.identity_block(x, init_weights=init_weights, **metaparameters)
        return x

    @staticmethod
    def identity_block(x, init_weights=None, **metaparameters):
        """ Construct a B(3,3) style block
            x        : input into the block
            n_filters: number of filters
            k        : width factor
            dropout  : dropout rate
            reg      : kernel regularizer
        """
        n_filters = metaparameters['n_filters']
        k         = metaparameters['k']
        dropout   = metaparameters['dropout']
        if 'reg' in metaparameters:
            reg = metaparameters['reg']
        else:
            reg = WRN.reg

        if init_weights is None:
            init_weights = WRN.init_weights
    
        # Save input vector (feature maps) for the identity link
        shortcut = x
    
        ## Construct the 3x3, 3x3 convolution block
    
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(n_filters * k, (3, 3), strides=(1, 1), padding='same', use_bias=False, 
                   kernel_initializer=init_weights, kernel_regularizer=reg)(x)

        # dropout only in identity link (not projection)
        if dropout > 0:
            x = Dropout(dropout)

        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(n_filters * k, (3, 3), strides=(1, 1), padding='same', use_bias=False, 
                   kernel_initializer=init_weights, kernel_regularizer=reg)(x)

        # Add the identity link (input) to the output of the residual block
        x = Add()([shortcut, x])
        return x

    @staticmethod
    def projection_block(x, init_weights=None, **metaparameters):
        """ Construct a B(3,3) style block
            x        : input into the block
            n_filters: number of filters
            k        : width factor
            strides  : whether the projection shortcut is strided
            reg      : kernel regularizer
        """
        n_filters = metaparameters['n_filters']
        strides   = metaparameters['strides']
        k         = metaparameters['k']
        if 'reg' in metaparameters:
            reg = metaparameters['reg']
        else:
            reg = WRN.reg

        if init_weights is None:
            init_weights = WRN.init_weights
   
        # Save input vector (feature maps) for the identity link
        shortcut = BatchNormalization()(x)
        shortcut = Conv2D(n_filters *k, (3, 3), strides=strides, padding='same', use_bias=False,
                          kernel_initializer=init_weights, kernel_regularizer=reg)(shortcut)
   
        ## Construct the 3x3, 3x3 convolution block
   
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(n_filters * k, (3, 3), strides=strides, padding='same', use_bias=False,
                   kernel_initializer=init_weights, kernel_regularizer=reg)(x)

        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(n_filters * k, (3, 3), strides=(1, 1), padding='same', use_bias=False,
                   kernel_initializer=init_weights, kernel_regularizer=reg)(x)

        # Add the identity link (input) to the output of the residual block
        x = Add()([shortcut, x])
        return x

    def classifier(self, x, n_classes, **metaparameters):
        """ Construct the Classifier Group 
            x         : input to the classifier
            n_classes : number of output classes
        """
        reg = metaparameters['reg']

        # Pool at the end of all the convolutional residual blocks (8, 8)
        x = AveragePooling2D((x.shape[1], x.shape[2]))(x)
        x = Flatten()(x)

        # Final Dense Outputting Layer for the outputs
        outputs = Dense(n_classes, activation='softmax', 
                        kernel_initializer=self.init_weights, kernel_regularizer=reg)(x)
        return outputs

# Example
# wrn = WRN(depth=28,k=10)
