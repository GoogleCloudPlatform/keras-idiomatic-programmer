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


# MobileNet v3 composable (2019)
# Paper: https://arxiv.org/pdf/1905.02244.pdf
# 224x224 input: 9,954,912 parameters (Large), 4,266,656 parameters (Small)

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D
from tensorflow.keras.layers import DepthwiseConv2D, Add, Reshape, Dense, Multiply, Activation
from tensorflow.keras.regularizers import l2

import sys
sys.path.append('../')
from models_c import Composable

class MobileNetV3(Composable):
    ReLU6 = Composable.ReLU
    HS    = Composable.HS
    
    """ Construct a Mobile Convolution Neural Network V3 """
    # Meta-parameter: number of filters/filter size, blocks per group, strides of projection block, activation, and
    #                 expansion per block
    groups = { 'large' : [ { 'n_filters' : 16,    'kernel_size': (3, 3), 'strides': (1, 1), 
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
                         ],

               'small' : [ { 'n_filters' : 16,    'kernel_size': (3, 3), 'strides': (2, 2),
                             'activation': ReLU6, 'blocks': [16], 'squeeze': True },
                           { 'n_filters' : 24,    'kernel_size': (3, 3), 'strides': (2, 2),
                             'activation': ReLU6, 'blocks': [72, 88], 'squeeze': False},
                           { 'n_filters' : 40,    'kernel_size': (5, 5), 'strides': (1, 1),
                             'activation': HS ,   'blocks': [96, 240, 240], 'squeeze': True},
                           { 'n_filters' : 48,    'kernel_size': (5, 5), 'strides': (2, 2),
                             'activation': HS ,   'blocks': [120, 144], 'squeeze': True},
                           { 'n_filters' : 96,    'kernel_size': (5, 5), 'n_blocks' : 3, 'strides': (2, 2),
                             'activation': HS ,   'blocks': [288, 576, 576], 'squeeze': True},
                           # Last block
                           { 'n_filters' : 576,   'kernel_size': (1, 1), 'strides': (1, 1),
                             'activation': HS ,   'blocks': None, 'squeeze': False}
                         ]
              }

    # Meta-parameter: width multiplier (0 .. 1) for reducing number of filters.
    alpha = 1
    # Meta-parameter: kernel regularization

    init_weights = 'glorot_uniform'
    relu = 6.0

    def __init__(self, groups, alpha=1, input_shape=(224, 224, 3), n_classes=1000,
                 init_weights='glorot_uniform', reg=l2(0.001), relu=6.0):
        """ Construct a Mobile Convolution Neural Network V3
            groups      : number of filters and blocks per group
            alpha       : width multiplier
            input_shape : the input shape
            n_classes   : number of output classes
            reg         : kernel regularizer
            init_weights: kernel initializer
            relu        : max value for ReLU
        """
        # Configure base (super) class
        super().__init__(init_weights=init_weights, reg=reg, relu=relu)
        
        # predefined
        if isinstance(groups, str):
            if groups not in ['large', 'small']:
                raise Exception("MobileNetV3: Invalid value for groups")
            groups = list(self.groups[groups])

        inputs = Input(shape=(224, 224, 3))

        # The Stem Group
        x = self.stem(inputs, alpha=alpha)

        # The Learner
        x = self.learner(x, groups=groups, alpha=alpha)

        # The Classifier 
        outputs = self.classifier(x, n_classes)

        # Instantiate the Model
        self._model = Model(inputs, outputs)

    def stem(self, inputs, **metaparameters):
        """ Construct the Stem Group
            inputs : input tensor
            alpha  : width multiplier
        """
        alpha = metaparameters['alpha']

        # Calculate the number of filters for the stem convolution
        # Must be divisible by 8
        n_filters = max(8, (int(32 * alpha) + 4) // 8 * 8)
    
        # Convolutional block
        x = Conv2D(n_filters, (3, 3), strides=(2, 2), padding='same', use_bias=False,
                   kernel_initializer=self.init_weights, kernel_regularizer=self.reg)(inputs)
        x = BatchNormalization()(x)
        x = Composable.HS(x)
        return x
    
    def learner(self, x, **metaparameters):
        """ Construct the Learner
            x        : input to the learner
            alpha    : width multiplier
        """
        groups = metaparameters['groups']
        alpha  = metaparameters['alpha']

        last = groups.pop()

        # Add Attention Residual Convolution Groups
        for group in groups:
            x = MobileNetV3.group(x, **group, **metaparameters)

        # Last block is a 1x1 linear convolutional layer,
        # expanding the number of filters to 1280.
        x = Conv2D(last['n_filters'] * alpha, (1, 1), strides=(1, 1), padding='same', use_bias=False,
                   kernel_initializer=self.init_weights, kernel_regularizer=self.reg)(x)
        x = BatchNormalization()(x)
        x = last['activation'](x)
        return x

    @staticmethod
    def group(x, **metaparameters):
        """ Construct an Attention Residual Group
            x         : input to the group
            blocks    : expansion per block
            strides   : whether first block uses strided convolution in project shortcut
        """   
        blocks  = metaparameters['blocks']
        strides   = metaparameters['strides']
        del metaparameters['strides']

        # In first block, the attention residual block maybe strided - feature map size reduction
        x = MobileNetV3.attention_block(x, strides=strides, expansion=blocks.pop(0), **metaparameters)
    
        # Remaining blocks
        for block in blocks:
            x = MobileNetV3.attention_block(x, strides=(1, 1), expansion=block, **metaparameters)
        return x

    @staticmethod
    def attention_block(x, strides=(1, 1), **metaparameters):
        """ Construct an Attention Residual Block
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
            alpha = MobileNetV3.alpha
        if 'reg' in metaparameters:
            reg = metaparameters['reg']
        else:
            reg = MobileNetV3.reg
        if 'init_weights' in metaparameters:
            init_weights = metaparameters['init_weights']
        else:
            init_weights = MobileNetV3.init_weights
        if 'squeeze' in metaparameters:
            squeeze = metaparameters['squeeze']
        else:
            squeeze = False
        if 'activation' in metaparameters:
            activation = metaparameters['activation']
        else:
            activation = ReLU6
            
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
            x = MobileNetV3.squeeze(x)

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
        x = Dense(n_channels, activation=Composable.ReLU)(x)
        x = Dense(n_channels, activation=Composable.HS)(x)
        x = Reshape((1, 1, n_channels))(x)
        x = Multiply()([shortcut, x])
        return x

    def classifier(self, x, n_classes):
        """ Construct the classifier group
            x         : input to the classifier
            n_classes : number of output classes
        """
        # Save encoding layer
        self.encoding = x

        # 7x7 Pooling
        n_channels = x.shape[-1]
        x = GlobalAveragePooling2D()(x)

        # Save embedding layer
        self.embedding = x
        
        x = Reshape((1, 1, n_channels))(x)

        x = Conv2D(1280, (1, 1), padding='same', activation=Composable.HS,
                   kernel_initializer=self.init_weights, kernel_regularizer=self.reg)(x)

        # final classification
        x = Conv2D(n_classes, (1, 1), padding='same', activation='softmax',
                   kernel_initializer=self.init_weights, kernel_regularizer=self.reg)(x)
        # Flatten the feature maps into 1D feature maps (?, N)
        outputs = Reshape((n_classes,))(x)

        # Save post-activation layer
        self.softmax = outputs

        return outputs

# Example
# mobilenet = MobileNetV3('large')
