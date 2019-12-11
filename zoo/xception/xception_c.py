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

    init_weights = 'glorot_uniform'

    def __init__(self, entry=None, middle=None, input_shape=(229, 229, 3), n_classes=1000,
                 init_weights=init_weights, reg=None, relu=None):
        """ Construct an Xception Convolution Neural Network
            entry       : number of blocks/filters for entry module
            middle      : number of blocks/filters for middle module
            input_shape : the input shape
            n_classes   : number of output classes
            init_weights: kernel initializer
            reg         : kernel regularizer
            relu        : max value for ReLU
        """
        super().__init__(init_weights=init_weights, reg=reg, relu=relu)
        
        if entry is None:
            entry = Xception.entry
        if middle is None:
            middle = Xception.middle

        # Create the input vector
        inputs = Input(shape=input_shape)

	# Create entry section with three blocks
        x = Xception.entryFlow(inputs, blocks=entry)

	# Create the middle section with eight blocks
        x = Xception.middleFlow(x, blocks=middle)

	# Create the exit section 
        outputs = Xception.exitFlow(x, n_classes)

	# Instantiate the model
        self._model = Model(inputs, outputs)

    @staticmethod
    def entryFlow(inputs, init_weights=None, **metaparameters):
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
            x = Composable.Conv2D(inputs, 32, (3, 3), strides=(2, 2), **metaparameters)
            x = BatchNormalization()(x)
            x = Composable.ReLU(x)

            # Convolution - dimensionality expansion
            # Double the number of filters
            x = Composable.Conv2D(x, 64, (3, 3), strides=(1, 1), **metaparameters)
            x = BatchNormalization()(x)
            x = Composable.ReLU(x)
            return x

        blocks = metaparameters['blocks']

        # Create the stem to the neural network
        x = stem(inputs)

        # Create residual blocks using linear projection
        for block in blocks:
            x = Xception.projection_block(x, **block)

        return x

    @staticmethod
    def middleFlow(x, **metaparameters):
        """ Create the middle flow section
            x     : input tensor into section
            blocks: number of filters per block
        """
        blocks = metaparameters['blocks']

        # Create residual blocks
        for block in blocks:
            x = Xception.residual_block(x, **block, **metaparameters)
        return x

    @staticmethod
    def exitFlow(x, n_classes, **metaparameters):
        """ Create the exit flow section
            x         : input to the exit flow section
            n_classes : number of output classes
        """     
        # Remember the input
        shortcut = x

        # Strided convolution to double number of filters in identity link to
        # match output of residual block for the add operation (projection shortcut)
        shortcut = Composable.Conv2D(x, 1024, (1, 1), strides=(2, 2), padding='same', **metaparameters)
        shortcut = BatchNormalization()(shortcut)

        # First Depthwise Separable Convolution
        # Dimensionality reduction - reduce number of filters
        x = Composable.SeparableConv2D(x, 728, (3, 3), padding='same', **metaparameters)
        x = BatchNormalization()(x)

        # Second Depthwise Separable Convolution
        # Dimensionality restoration
        x = Composable.SeparableConv2D(x, 1024, (3, 3), padding='same', **metaparameters)
        x = BatchNormalization()(x)
        x = Composable.ReLU(x)

        # Create pooled feature maps, reduce size by 75%
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

        # Add the projection shortcut to the output of the pooling layer
        x = Add()([x, shortcut])

        # Third Depthwise Separable Convolution
        x = Composable.SeparableConv2D(x, 1556, (3, 3), padding='same', **metaparameters)
        x = BatchNormalization()(x)
        x = Composable.ReLU(x)

        # Fourth Depthwise Separable Convolution
        x = Composable.SeparableConv2D(x, 2048, (3, 3), padding='same', **metaparameters)
        x = BatchNormalization()(x)
        x = Composable.ReLU(x)

        # Create classifier section
        x = Composable.classifier(x, n_classes, **metaparameters)

        return x

    @staticmethod
    def projection_block(x, **metaparameters):
        """ Create a residual block using Depthwise Separable Convolutions with Projection shortcut
            x        : input into residual block
            n_filters: number of filters
        """
        n_filters = metaparameters['n_filters']
        del metaparameters['n_filters']

        # Remember the input
        shortcut = x
    
        # Strided convolution to double number of filters in identity link to
        # match output of residual block for the add operation (projection shortcut)
        shortcut = Composable.Conv2D(x, n_filters, (1, 1), strides=(2, 2), padding='same', **metaparameters)
        shortcut = BatchNormalization()(shortcut)

        # First Depthwise Separable Convolution
        x = Composable.SeparableConv2D(x, n_filters, (3, 3), padding='same', **metaparameters)
        x = BatchNormalization()(x)
        x = Composable.ReLU(x)

        # Second depthwise Separable Convolution
        x = Composable.SeparableConv2D(x, n_filters, (3, 3), padding='same', **metaparameters)
        x = BatchNormalization()(x)
        x = Composable.ReLU(x)

        # Create pooled feature maps, reduce size by 75%
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

        # Add the projection shortcut to the output of the block
        x = Add()([x, shortcut])

        return x

    @staticmethod
    def residual_block(x, **metaparameters):
        """ Create a residual block using Depthwise Separable Convolutions
            x        : input into residual block
            n_filters: number of filters
        """
        n_filters = metaparameters['n_filters']
        del metaparameters['n_filters']

        # Remember the input
        shortcut = x

        # First Depthwise Separable Convolution
        x = Composable.SeparableConv2D(x, n_filters, (3, 3), padding='same', **metaparameters)
        x = BatchNormalization()(x)
        x = Composable.ReLU(x)

        # Second depthwise Separable Convolution
        x = Composable.SeparableConv2D(x, n_filters, (3, 3), padding='same', **metaparameters)
        x = BatchNormalization()(x)
        x = Composable.ReLU(x)

        # Third depthwise Separable Convolution
        x = Composable.SeparableConv2D(x, n_filters, (3, 3), padding='same', **metaparameters)
        x = BatchNormalization()(x)
        x = Composable.ReLU(x)
    
        # Add the identity link to the output of the block
        x = Add()([x, shortcut])
        return x

# Example
# xception = Xception()
