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


# MobileNet v3 (Large) + composable (2019)
# Paper: https://arxiv.org/pdf/1905.02244.pdf
# 224x224 input: 9,954,912 parameters

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D
from tensorflow.keras.layers import DepthwiseConv2D, Add, Reshape, Dense, Multiply
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K

def ReLU6(x):
    """ ReLU activation clipped at 6 """
    return ReLU(6.0)(x)

def HS(x):
    """ Hard Swish activation """
    return (x * K.relu(x + 3, max_value=6.0)) / 6.0

class MobileNetV3Large(object):
    """ Construct a Mobile Convolution Neural Network """
    # Meta-parameter: number of filters/filter size, blocks per group, strides of projection block, activation, and
    #                 expansion per block
    groups = [ { 'n_filters' : 16,    'kernel_size': (3, 3), 'strides': (1, 1), 
                 'activation': ReLU6, 'blocks': [16], 'squeeze': False }, 
               { 'n_filters' : 24,    'kernel_size': (3, 3), 'strides': (2, 2), 
                 'activation': ReLU6, 'blocks': [64, 72], 'squeeze': False},
               { 'n_filters' : 40,    'kernel_size': (5, 5), 'strides': (2, 2), 
                 'activation': ReLU6, 'blocks': [72, 120, 120], 'squeeze': True}, 
               { 'n_filters' : 80,    'kernel_size': (5, 5), 'strides': (2, 2), 
                 'activation': HS ,   'blocks': [240, 200, 184, 184], 'squeeze': False},
               { 'n_filters' : 112,   'kernel_size': (5, 5), 'strides': (1, 1),
                 'activation': HS ,   'blocks': [480, 672], 'squeeze': True},
               { 'n_filters' : 160,   'kernel_size': (5, 5), 'strides': (2, 2),
                 'activation': HS ,   'blocks': [672, 960, 960], 'squeeze': True},
               # Last block
               { 'n_filters' : 960,   'kernel_size': (1, 1), 'strides': (1, 1), 
                 'activation': HS ,   'blocks': None, 'squeeze': False}
             ]

    # Meta-parameter: width multiplier (0 .. 1) for reducing number of filters.
    alpha = 1
    # Meta-parameter: kernel regularization
    reg = l2(0.001)
    init_weights = 'glorot_uniform'
    _model = None

    def __init__(self, groups=None, alpha=1, reg=l2(0.001), input_shape=(224, 224, 3), n_classes=1000):
        """ Construct a Mobile Convolution Neural Network
            groups     : number of filters and blocks per group
            alpha      : width multiplier
            reg        : kernel regularizer
            input_shape: the input shape
            n_classes  : number of output classes
        """
        if groups is None:
             groups = self.groups

        inputs = Input(shape=(224, 224, 3))

        # The Stem Group
        x = self.stem(inputs, alpha=alpha, reg=reg)

        # The Learner
        x = self.learner(x, groups=groups, alpha=alpha, reg=reg)

        # The Classifier 
        outputs = self.classifier(x, n_classes, reg=reg)

        # Instantiate the Model
        self._model = Model(inputs, outputs)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, _model):
        self._model = model

    def stem(self, inputs, **metaparameters):
        """ Construct the Stem Group
            inputs : input tensor
            alpha  : width multiplier
            reg    : kernel regularizer
        """
        alpha = metaparameters['alpha']
        reg   = metaparameters['reg']

        # Calculate the number of filters for the stem convolution
        # Must be divisible by 8
        n_filters = max(8, (int(32 * alpha) + 4) // 8 * 8)
    
        # Convolutional block
        x = Conv2D(n_filters, (3, 3), strides=(2, 2), padding='same', use_bias=False,
                   kernel_initializer=self.init_weights, kernel_regularizer=reg)(inputs)
        x = BatchNormalization()(x)
        x = HS(x)
        return x
    
    def learner(self, x, **metaparameters):
        """ Construct the Learner
            x        : input to the learner
            alpha    : width multiplier
            reg      : kernel regularizer
        """
        groups = metaparameters['groups']
        alpha  = metaparameters['alpha']
        reg    = metaparameters['reg']

        last = groups.pop()

        # Add Inverted Residual Convolution Groups
        for group in groups:
            x = MobileNetV3Large.group(x, **group, **metaparameters)

        # Last block is a 1x1 linear convolutional layer,
        # expanding the number of filters to 1280.
        x = Conv2D(last['n_filters'] * alpha, (1, 1), strides=(1, 1), padding='same', use_bias=False,
                   kernel_initializer=self.init_weights, kernel_regularizer=reg)(x)
        x = BatchNormalization()(x)
        x = last['activation'](x)
        return x

    @staticmethod
    def group(x, **metaparameters):
        """ Construct an Inverted Residual Group
            x         : input to the group
            blocks    : expansion per block
            strides   : whether first block uses strided convolution in project shortcut
        """   
        blocks  = metaparameters['blocks']
        strides   = metaparameters['strides']
        del metaparameters['strides']

        # In first block, the inverted residual block maybe strided - feature map size reduction
        x = MobileNetV3Large.inverted_block(x, strides=strides, expansion=blocks.pop(0), **metaparameters)
    
        # Remaining blocks
        for block in blocks:
            x = MobileNetV3Large.inverted_block(x, strides=(1, 1), expansion=block, **metaparameters)
        return x

    @staticmethod
    def inverted_block(x, strides=(1, 1), init_weights=None, **metaparameters):
        """ Construct an Inverted Residual Block
            x         : input to the block
            strides   : strides
            n_filters : number of filters
            alpha     : width multiplier
            expansion : multiplier for expanding number of filters
            squeeze   : whether to include squeeze
            activation: type of activation function
            reg       : kernel regularizer
        """
        n_filters = metaparameters['n_filters']
        expansion = metaparameters['expansion']
        alpha     = metaparameters['alpha']
        if 'alpha' in metaparameters:
            alpha = metaparameters['alpha']
        else:
            alpha = MobileNetV3Large.alpha
        if 'reg' in metaparameters:
            reg = metaparameters['reg']
        else:
            reg = MobileNetV3Large.reg
        if 'squeeze' in metaparameters:
            squeeze = metaparameters['squeeze']
        else:
            squeeze = False
        if 'activation' in metaparameters:
            activation = metaparameters['activation']
        else:
            activation = ReLU6

        if init_weights is None:
            init_weights = MobileNetV3Large.init_weights
            
        # Remember input
        shortcut = x

        # Apply the width filter to the number of feature maps for the pointwise convolution
        filters = int(n_filters * alpha)
    
        n_channels = int(x.shape[3])
    
        # Dimensionality Expansion
        # 1x1 linear convolution
        x = Conv2D(expansion, (1, 1), padding='same', use_bias=False,
                       kernel_initializer=init_weights, kernel_regularizer=reg)(x)
        x = BatchNormalization()(x)
        x = activation(x)

        # Depthwise Convolution
        x = DepthwiseConv2D((3, 3), strides, padding='same', use_bias=False,
                            kernel_initializer=init_weights, kernel_regularizer=reg)(x)
        x = BatchNormalization()(x)
        x = activation(x)

        # Add squeeze (dimensionality reduction)
        if squeeze:
            x = MobileNetV3Large.squeeze(x)

        # Linear Pointwise Convolution
        x = Conv2D(filters, (1, 1), strides=(1, 1), padding='same', use_bias=False,
                   kernel_initializer=init_weights, kernel_regularizer=reg)(x)
        x = BatchNormalization()(x)
    
        # Number of input filters matches the number of output filters
        if n_channels == filters and strides == (1, 1):
            x = Add()([shortcut, x]) 
        return x

    @staticmethod
    def squeeze(x):
        """ Construct a squeeze block
            x   : input to the squeeze
        """
        shortcut = x
        n_channels = x.shape[-1]

        x = GlobalAveragePooling2D()(x)
        x = Dense(n_channels, activation=ReLU6)(x)
        x = Dense(n_channels, activation=HS)(x)
        x = Reshape((1, 1, n_channels))(x)
        x = Multiply()([shortcut, x])
        return x

    def classifier(self, x, n_classes, **metaparameters):
        """ Construct the classifier group
            x         : input to the classifier
            n_classes : number of output classes
            reg       : kernel regularizer
        """
        reg = metaparameters['reg']

        # 7x7 Pooling
        n_channels = x.shape[-1]
        x = GlobalAveragePooling2D()(x)
        x = Reshape((1, 1, n_channels))(x)

        x = Conv2D(1280, (1, 1), padding='same', activation=HS,
                   kernel_initializer=self.init_weights, kernel_regularizer=reg)(x)

        # final classification
        x = Conv2D(n_classes, (1, 1), padding='same', activation='softmax',
                   kernel_initializer=self.init_weights, kernel_regularizer=reg)(x)
        # Flatten the feature maps into 1D feature maps (?, N)
        x = Reshape((n_classes,))(x)

        return x

# Example
# mobilenet = MobileNetV3Large()
