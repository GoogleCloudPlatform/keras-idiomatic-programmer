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
# https://arxiv.org/pdf/1610.02357.pdf

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Dense, GlobalAveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, MaxPooling2D, Add
from tensorflow.keras.regularizers import l2

class Xception(object):
    """ Construct an Xception Convolution Neural Network """
    # meta-parameter: number of filters per block
    entry  = [{ 'n_filters' : 128 }, { 'n_filters' : 256 }, { 'n_filters' : 728 }]
    middle = [{ 'n_filters' : 728 }, { 'n_filters' : 728 }, { 'n_filters' : 728 }, 
              { 'n_filters' : 728 }, { 'n_filters' : 728 }, { 'n_filters' : 728 }, 
              { 'n_filters' : 728 }, { 'n_filters' : 728 } ]

    # meta-parameter: kernel regularizer
    reg = l2(0.001)

    init_weights = 'glorot_uniform'
    _model = None

    def __init__(self, entry=None, middle=None, reg=l2(0.001), input_shape=(229, 229, 3), n_classes=1000):
        """ Construct an Xception Convolution Neural Network
            entry      : number of blocks/filters for entry module
            middle     : number of blocks/filters for middle module
            reg        : kernel regularizer
            input_shape: the input shape
            n_classes  : number of output classes
        """
        if entry is None:
            entry = Xception.entry
        if middle is None:
            middle = Xception.middle

        # Create the input vector
        inputs = Input(shape=input_shape)

	# Create entry section with three blocks
        x = Xception.entryFlow(inputs, blocks=entry, reg=reg)

	# Create the middle section with eight blocks
        x = Xception.middleFlow(x, blocks=middle, reg=reg)

	# Create the exit section 
        outputs = Xception.exitFlow(x, n_classes, reg=reg)

	# Instantiate the model
        self._model = Model(inputs, outputs)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, _model):
        return self._model

    @staticmethod
    def entryFlow(inputs, init_weights=None, **metaparameters):
        """ Create the entry flow section
            inputs : input tensor to neural network
            blocks : number of filters per block
            reg    : kernel nregularizer
        """
        if 'reg' in metaparameters:
            reg = metaparameters['reg']
        else:
            reg = Xception.reg

        def stem(inputs, init_weights):
            """ Create the stem entry into the neural network
                inputs : input tensor to neural network
            """
            # Strided convolution - dimensionality reduction
            # Reduce feature maps by 75%
            x = Conv2D(32, (3, 3), strides=(2, 2),
                       kernel_initializer=init_weights, kernel_regularizer=reg)(inputs)
            x = BatchNormalization()(x)
            x = ReLU()(x)

            # Convolution - dimensionality expansion
            # Double the number of filters
            x = Conv2D(64, (3, 3), strides=(1, 1),
                       kernel_initializer=init_weights, kernel_regularizer=reg)(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            return x

        blocks = metaparameters['blocks']

        if init_weights is None:
            init_weights = Xception.init_weights

        # Create the stem to the neural network
        x = stem(inputs, init_weights)

        # Create residual blocks using linear projection
        for block in blocks:
            x = Xception.projection_block(x, init_weights, **block, reg=reg)

        return x

    @staticmethod
    def middleFlow(x, init_weights=None, **metaparameters):
        """ Create the middle flow section
            x     : input tensor into section
            blocks: number of filters per block
        """
        blocks = metaparameters['blocks']

        if init_weights is None:
            init_weights = Xception.init_weights

        # Create residual blocks
        for block in blocks:
            x = Xception.residual_block(x, init_weights, **block, **metaparameters)
        return x

    @staticmethod
    def exitFlow(x, n_classes, init_weights=None, **metaparameters):
        """ Create the exit flow section
            x         : input to the exit flow section
            n_classes : number of output classes
            reg       : kernel regularizer
        """
        if 'reg' in metaparameters:
            reg = metaparameters['reg']
        else:
            reg = Xception.reg
            
        def classifier(x, n_classes, init_weights):
            """ The output classifier
                x         : input to the classifier
                n_classes : number of output classes
            """
            # Global Average Pooling will flatten the 10x10 feature maps into 1D feature maps
            x = GlobalAveragePooling2D()(x)
        
            # Fully connected output layer (classification)
            x = Dense(n_classes, activation='softmax',
                    kernel_initializer=init_weights, kernel_regularizer=reg)(x)
            return x

        if init_weights is None:
            init_weights = Xception.init_weights

        # Remember the input
        shortcut = x

        # Strided convolution to double number of filters in identity link to
        # match output of residual block for the add operation (projection shortcut)
        shortcut = Conv2D(1024, (1, 1), strides=(2, 2), padding='same',
                          kernel_initializer=init_weights)(shortcut)
        shortcut = BatchNormalization()(shortcut)

        # First Depthwise Separable Convolution
        # Dimensionality reduction - reduce number of filters
        x = SeparableConv2D(728, (3, 3), padding='same',
                            kernel_initializer=init_weights, kernel_regularizer=reg)(x)
        x = BatchNormalization()(x)

        # Second Depthwise Separable Convolution
        # Dimensionality restoration
        x = SeparableConv2D(1024, (3, 3), padding='same',
                            kernel_initializer=init_weights, kernel_regularizer=reg)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        # Create pooled feature maps, reduce size by 75%
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

        # Add the projection shortcut to the output of the pooling layer
        x = Add()([x, shortcut])

        # Third Depthwise Separable Convolution
        x = SeparableConv2D(1556, (3, 3), padding='same',
                            kernel_initializer=init_weights, kernel_regularizer=reg)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        # Fourth Depthwise Separable Convolution
        x = SeparableConv2D(2048, (3, 3), padding='same',
                            kernel_initializer=init_weights, kernel_regularizer=reg)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        # Create classifier section
        x = classifier(x, n_classes, init_weights)

        return x

    @staticmethod
    def projection_block(x, init_weights=None, **metaparameters):
        """ Create a residual block using Depthwise Separable Convolutions with Projection shortcut
            x        : input into residual block
            n_filters: number of filters
            reg       : kernel regularizer
        """
        n_filters = metaparameters['n_filters']
        if 'reg' in metaparameters:
            reg = metaparameters['reg']
        else:
            reg = Xception.reg

        if init_weights is None:
            init_weights = Xception.init_weights

        # Remember the input
        shortcut = x
    
        # Strided convolution to double number of filters in identity link to
        # match output of residual block for the add operation (projection shortcut)
        shortcut = Conv2D(n_filters, (1, 1), strides=(2, 2), padding='same', kernel_initializer=init_weights)(shortcut)
        shortcut = BatchNormalization()(shortcut)

        # First Depthwise Separable Convolution
        x = SeparableConv2D(n_filters, (3, 3), padding='same', kernel_initializer=init_weights)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        # Second depthwise Separable Convolution
        x = SeparableConv2D(n_filters, (3, 3), padding='same', kernel_initializer=init_weights)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        # Create pooled feature maps, reduce size by 75%
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

        # Add the projection shortcut to the output of the block
        x = Add()([x, shortcut])

        return x

    @staticmethod
    def residual_block(x, init_weights=None, **metaparameters):
        """ Create a residual block using Depthwise Separable Convolutions
            x        : input into residual block
            n_filters: number of filters
            reg       : kernel regularizer
        """
        n_filters = metaparameters['n_filters']
        if 'reg' in metaparameters:
            reg = metaparameters['reg']
        else:
            reg = Xception.reg

        if init_weights is None:
            init_weights = Xception.init_weights

        # Remember the input
        shortcut = x

        # First Depthwise Separable Convolution
        x = SeparableConv2D(n_filters, (3, 3), padding='same',
                            kernel_initializer=init_weights, kernel_regularizer=reg)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        # Second depthwise Separable Convolution
        x = SeparableConv2D(n_filters, (3, 3), padding='same',
                            kernel_initializer=init_weights, kernel_regularizer=reg)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        # Third depthwise Separable Convolution
        x = SeparableConv2D(n_filters, (3, 3), padding='same',
                            kernel_initializer=init_weights, kernel_regularizer=reg)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
    
        # Add the identity link to the output of the block
        x = Add()([x, shortcut])
        return x

# Example
# xception = Xception()
