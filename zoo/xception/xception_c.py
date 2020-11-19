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

# Xception + Composable (2016)
# Trainable params: 22,944,896
# https://arxiv.org/pdf/1610.02357.pdf

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Dense, GlobalAveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, MaxPooling2D, Add, Activation
from tensorflow.keras.regularizers import l2

import sys
sys.path.append('../')
from models_c import Composable

class Xception(Composable):
    """ Construct an Xception Convolution Neural Network """
    # meta-parameter: number of filters per block
    entry  = [{ 'n_filters' : 128 }, { 'n_filters' : 256 }, { 'n_filters' : 728 }]
    middle = [{ 'n_filters' : 728 }, { 'n_filters' : 728 }, { 'n_filters' : 728 }, 
              { 'n_filters' : 728 }, { 'n_filters' : 728 }, { 'n_filters' : 728 }, 
              { 'n_filters' : 728 }, { 'n_filters' : 728 } ]

    # Initial Hyperparameters
    hyperparameters = { 'initializer': 'glorot_uniform',
                        'regularizer': None,
                        'relu_clip'  : None,
                        'bn_epsilon' : None,
                        'use_bias'   : True
                      }

    def __init__(self, entry=None, middle=None, 
                 input_shape=(229, 229, 3), n_classes=1000, include_top=True,
                 **hyperparameters):
        """ Construct an Xception Convolution Neural Network
            entry       : number of blocks/filters for entry module
            middle      : number of blocks/filters for middle module
            input_shape : the input shape
            n_classes   : number of output classes
            include_top : whether to include classifier
            initializer : kernel initializer
            regularizer : kernel regularizer
            relu_clip   : max value for ReLU
            bn_epsilon  : epsilon for batch norm
            use_bias    : whether to use bias in conjunction with batch norm
        """
        Composable.__init__(self, input_shape, include_top, self.hyperparameters, **hyperparameters)
        
        if entry is None:
            entry = self.entry
        if middle is None:
            middle = self.middle

        # Create the input vector
        inputs = Input(shape=input_shape)

	# Create entry section with three blocks
        x = self.entryFlow(inputs, blocks=entry)

	# Create the middle section with eight blocks
        x = self.middleFlow(x, blocks=middle)

	# Create the exit section 
        outputs = self.exitFlow(x, n_classes, include_top)

	# Instantiate the model
        self._model = Model(inputs, outputs)

    def entryFlow(self, inputs, init_weights=None, **metaparameters):
        """ Create the entry flow section
            inputs : input tensor to neural network
            blocks : number of filters per block
        """

        def stem(inputs):
            """ Create the stem entry into the neural network
                inputs : input tensor to neural network
            """
            # Strided convolution - dimensionality reduction
            # Reduce feature maps by 75%
            x = self.Conv2D(inputs, 32, (3, 3), strides=(2, 2), **metaparameters)
            x = self.BatchNormalization(x)
            x = self.ReLU(x)

            # Convolution - dimensionality expansion
            # Double the number of filters
            x = self.Conv2D(x, 64, (3, 3), strides=(1, 1), **metaparameters)
            x = self.BatchNormalization(x)
            x = self.ReLU(x)
            return x

        blocks = metaparameters['blocks']

        # Create the stem to the neural network
        x = stem(inputs)

        # Create residual blocks using linear projection
        for block in blocks:
            x = self.projection_block(x, **block)

        return x

    def middleFlow(self, x, **metaparameters):
        """ Create the middle flow section
            x     : input tensor into section
            blocks: number of filters per block
        """
        blocks = metaparameters['blocks']

        # Create residual blocks
        for block in blocks:
            x = self.residual_block(x, **block, **metaparameters)
        return x

    def exitFlow(self, x, n_classes, include_top, **metaparameters):
        """ Create the exit flow section
            x          : input to the exit flow section
            n_classes  : number of output classes
            include_top: whether to include classifier
        """     
        # Remember the input
        shortcut = x

        # Strided convolution to double number of filters in identity link to
        # match output of residual block for the add operation (projection shortcut)
        shortcut = self.Conv2D(x, 1024, (1, 1), strides=(2, 2), padding='same', **metaparameters)
        shortcut = self.BatchNormalization(shortcut)

        # First Depthwise Separable Convolution
        # Dimensionality reduction - reduce number of filters
        x = self.SeparableConv2D(x, 728, (3, 3), padding='same', **metaparameters)
        x = self.BatchNormalization(x)
        x = self.ReLU(x)

        # Second Depthwise Separable Convolution
        # Dimensionality restoration
        x = self.SeparableConv2D(x, 1024, (3, 3), padding='same', **metaparameters)
        x = self.BatchNormalization(x)
        x = self.ReLU(x)

        # Create pooled feature maps, reduce size by 75%
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

        # Add the projection shortcut to the output of the pooling layer
        x = Add()([x, shortcut])

        # Third Depthwise Separable Convolution
        x = self.SeparableConv2D(x, 1556, (3, 3), padding='same', **metaparameters)
        x = self.BatchNormalization(x)
        x = self.ReLU(x)

        # Fourth Depthwise Separable Convolution
        x = self.SeparableConv2D(x, 2048, (3, 3), padding='same', **metaparameters)
        x = self.BatchNormalization(x)
        x = self.ReLU(x)

        # Create classifier section
        if include_top:
            x = self.classifier(x, n_classes, **metaparameters)

        return x

    def projection_block(self, x, **metaparameters):
        """ Create a residual block using Depthwise Separable Convolutions with Projection shortcut
            x        : input into residual block
            n_filters: number of filters
        """
        n_filters = metaparameters['n_filters']
        del metaparameters['n_filters']

        # 1x1 strided convolution to increase the number of and reduce size of the feature maps 
        # in identity link to match output of residual block for the add operation (projection shortcut)
        shortcut = self.Conv2D(x, n_filters, (1, 1), strides=(2, 2), padding='same', **metaparameters)
        shortcut = self.BatchNormalization(shortcut)

        # First Depthwise Separable Convolution
        x = self.SeparableConv2D(x, n_filters, (3, 3), padding='same', **metaparameters)
        x = self.BatchNormalization(x)
        x = self.ReLU(x)

        # Second depthwise Separable Convolution
        x = self.SeparableConv2D(x, n_filters, (3, 3), padding='same', **metaparameters)
        x = self.BatchNormalization(x)
        x = self.ReLU(x)

        # Create pooled feature maps, reduce size by 75%
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

        # Add the projection shortcut to the output of the block
        x = Add()([x, shortcut])

        return x

    def residual_block(self, x, **metaparameters):
        """ Create a residual block using Depthwise Separable Convolutions
            x        : input into residual block
            n_filters: number of filters
        """
        n_filters = metaparameters['n_filters']
        del metaparameters['n_filters']

        # Remember the input
        shortcut = x

        # First Depthwise Separable Convolution
        x = self.SeparableConv2D(x, n_filters, (3, 3), padding='same', **metaparameters)
        x = self.BatchNormalization(x)
        x = self.ReLU(x)

        # Second depthwise Separable Convolution
        x = self.SeparableConv2D(x, n_filters, (3, 3), padding='same', **metaparameters)
        x = self.BatchNormalization(x)
        x = self.ReLU(x)

        # Third depthwise Separable Convolution
        x = self.SeparableConv2D(x, n_filters, (3, 3), padding='same', **metaparameters)
        x = self.BatchNormalization(x)
        x = self.ReLU(x)
    
        # Add the identity link to the output of the block
        x = Add()([x, shortcut])
        return x

# Example
# xception = Xception()

def example():
    ''' Example for constructing/training a Xception model on CIFAR-10
    '''
    # Example of constructing a mini-Xception
    entry  = [{ 'n_filters' : 128 }, { 'n_filters' : 728 }]
    middle = [{ 'n_filters' : 728 }, { 'n_filters' : 728 }, { 'n_filters' : 728 }]

    xception = Xception(entry=entry, middle=middle, input_shape=(32, 32, 3), n_classes=10)
    xception.model.summary()
    xception.cifar10()

example()
