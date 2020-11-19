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
# Trainable params: 38,547,418
# Paper: https://arxiv.org/pdf/1605.07146.pdf 

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, Flatten, BatchNormalization, ReLU, Dropout
from tensorflow.keras.layers import Dense, AveragePooling2D, Add, MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.regularizers import l2

import sys
sys.path.append('../')
from models_c import Composable

class WRN(Composable):
    """ Construct a Wide Residual Convolution Network """
    # Meta-parameter: number of filters per group
    groups = [ { 'n_filters': 16 }, { 'n_filters' : 32 }, { 'n_filters' : 64 } ]

    # Initial Hyperparameters
    hyperparameters = { 'initializer': 'he_normal',
                        'regularizer': None,
                        'relu_clip'  : None,
                        'bn_epsilon' : None,
                        'use_bias'   : False
                      }

    def __init__(self, groups=None, depth=16, k=8, dropout=0, 
                 input_shape=(32, 32, 3), n_classes=10, include_top=True,
                 **hyperparameters):
        """ Construct a Wide Residual (Convolutional Neural) Network 
            depth       : number of layers
            k           : width factor
            groups      : number of filters per group
            input_shape : input shape
            n_classes   : number of output classes
            include_top : whether to include classifier
            initializer : kernel initialization
            regularizer : kernel regularization
            relu_clip   : max value for ReLU
            bn_epsilon  : epsilon for batch norm
            use_bias    : whether use bias in conjunction with batch norm
        """
        # Configure base (super) class
        Composable.__init__(self, input_shape, include_top, self.hyperparameters, **hyperparameters)

        if groups is None:
            groups = list(self.groups)

        # The input tensor
        inputs = Input(input_shape)

        # The stem convolutional group
        x = self.stem(inputs)

        # The learner
        outputs = self.learner(x, groups=groups, depth=depth, k=k, dropout=dropout)

        # The classifier 
        if include_top:
            # Add hidden dropout
            outputs = self.classifier(outputs, n_classes, dropout=0.0)

        # Instantiate the Model
        self._model = Model(inputs, outputs)

    def stem(self, inputs):
        """ Construct the Stem Convolutional Group 
            inputs : the input vector
        """
        # Convolutional layer 
        x = self.Conv2D(inputs, 16, (3, 3), strides=(1, 1), padding='same')
        x = self.BatchNormalization(x)
        x = self.ReLU(x)
    
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
        x = self.group(x, n_blocks=n_blocks, strides=(1, 1), **groups.pop(0), **metaparameters)
        
        # remaining groups
        for group in groups:
            x = self.group(x, n_blocks=n_blocks, strides=(2, 2), **group, **metaparameters)
        return x
    
    def group(self, x, **metaparameters):
        """ Construct a Wide Residual Group
            x         : input into the group
            n_blocks  : number of residual blocks with identity link
        """
        n_blocks  = metaparameters['n_blocks']

        # first block is projection to match the number of input filters to output fitlers for the add operation
        x = self.projection_block(x, **metaparameters)

        # wide residual blocks
        for _ in range(n_blocks-1):
            x = self.identity_block(x, **metaparameters)
        return x

    def identity_block(self, x, **metaparameters):
        """ Construct a B(3,3) style block
            x        : input into the block
            n_filters: number of filters
            k        : width factor
            dropout  : dropout rate
        """
        n_filters = metaparameters['n_filters']
        k         = metaparameters['k']
        dropout   = metaparameters['dropout']
        del metaparameters['n_filters']
        del metaparameters['strides']
    
        # Save input vector (feature maps) for the identity link
        shortcut = x
    
        ## Construct the 3x3, 3x3 convolution block
    
        x = self.BatchNormalization(x)
        x = self.ReLU(x)
        x = self.Conv2D(x, n_filters * k, (3, 3), strides=(1, 1), padding='same', **metaparameters)

        # dropout only in identity link (not projection)
        if dropout > 0:
            x = Dropout(dropout)

        x = self.BatchNormalization(x)
        x = self.ReLU(x)
        x = self.Conv2D(x, n_filters * k, (3, 3), strides=(1, 1), padding='same', **metaparameters)

        # Add the identity link (input) to the output of the residual block
        x = Add()([shortcut, x])
        return x

    def projection_block(self, x, **metaparameters):
        """ Construct a B(3,3) style block
            x        : input into the block
            n_filters: number of filters
            k        : width factor
            strides  : whether the projection shortcut is strided
        """
        n_filters = metaparameters['n_filters']
        strides   = metaparameters['strides']
        k         = metaparameters['k']
        del metaparameters['n_filters']
        del metaparameters['strides']
   
        # Save input vector (feature maps) for the identity link
        shortcut = self.BatchNormalization(x)
        shortcut = self.Conv2D(shortcut, n_filters *k, (3, 3), strides=strides, padding='same', **metaparameters)
   
        ## Construct the 3x3, 3x3 convolution block
   
        x = self.BatchNormalization(x)
        x = self.ReLU(x)
        x = self.Conv2D(x, n_filters * k, (3, 3), strides=strides, padding='same', **metaparameters)

        x = self.BatchNormalization(x)
        x = self.ReLU(x)
        x = self.Conv2D(x, n_filters * k, (3, 3), strides=(1, 1), padding='same', **metaparameters)

        # Add the identity link (input) to the output of the residual block
        x = Add()([shortcut, x])
        return x

# Example
# wrn = WRN(depth=28, k=10)

def example():
    ''' Example for constructing/training a WRN model on CIFAR-10
    '''
    # Example of constructing a mini WRN
    wrn = WRN(depth=14, k=2, input_shape=(32, 32, 3), n_classes=10)
    wrn.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
    wrn.model.summary()
    wrn.cifar10()

# example()
