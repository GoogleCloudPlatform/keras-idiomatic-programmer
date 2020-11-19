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
    """ Construct a Mobile Convolution Neural Network V3 """

    # Meta-parameter: number of filters/filter size, blocks per group, strides of projection block, activation, and
    #                 expansion per block
    def GROUPS(self):
        ReLU6 = self.ReLU
        HS    = self.HS
        self.groups = { 'large' : [ { 'n_filters' : 16,    'kernel_size': (3, 3), 'strides': (1, 1), 
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

    # Initial Hyperparameters
    hyperparameters = { 'initializer': 'glorot_uniform',
                        'regularizer': l2(0.001),
                        'relu_clip'  : 6.0,
                        'bn_epsilon' : None,
                        'use_bias'   : False
                      }

    def __init__(self, groups, alpha=1, 
                 input_shape=(224, 224, 3), n_classes=1000, include_top=True,
                 **hyperparameters):
        """ Construct a Mobile Convolution Neural Network V3
            groups      : number of filters and blocks per group
            alpha       : width multiplier
            input_shape : the input shape
            n_classes   : number of output classes
            include_top : whether to include classifier
            regularizer : kernel regularizer
            initializer : kernel initializer
            relu_clip   : max value for ReLU
            bn_epsilon  : epsilon for batch norm
            use_bias    : whether to use bias
        """
        # Configure base (super) class
        Composable.__init__(self, input_shape, include_top, self.hyperparameters, **hyperparameters)

        # Variable Binding
        self.GROUPS()
        
        # predefined
        if isinstance(groups, str):
            if groups not in ['large', 'small']:
                raise Exception("MobileNetV3: Invalid value for groups")
            groups = list(self.groups[groups])

        inputs = Input(shape=input_shape)

        # The Stem Group
        x = self.stem(inputs, alpha=alpha)

        # The Learner
        outputs = self.learner(x, groups=groups, alpha=alpha)

        # The Classifier 
        if include_top:
            outputs = self.classifier(outputs, n_classes)

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
        x = self.Conv2D(inputs, n_filters, (3, 3), strides=(2, 2), padding='same', **metaparameters)
        x = self.BatchNormalization(x)
        x = self.HS(x)
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
            x = self.group(x, **group, **metaparameters)

        # Last block is a 1x1 linear convolutional layer,
        # expanding the number of filters to 1280.
        x = self.Conv2D(x, last['n_filters'] * alpha, (1, 1), strides=(1, 1), padding='same', **metaparameters)
        x = self.BatchNormalization(x)
        x = last['activation'](x)
        return x

    def group(self, x, **metaparameters):
        """ Construct an Attention Residual Group
            x         : input to the group
            blocks    : expansion per block
            strides   : whether first block uses strided convolution in project shortcut
        """   
        blocks  = metaparameters['blocks']
        strides   = metaparameters['strides']
        del metaparameters['strides']

        # In first block, the attention residual block maybe strided - feature map size reduction
        x = self.attention_block(x, strides=strides, expansion=blocks.pop(0), **metaparameters)
    
        # Remaining blocks
        for block in blocks:
            x = self.attention_block(x, strides=(1, 1), expansion=block, **metaparameters)
        return x

    def attention_block(self, x, strides=(1, 1), **metaparameters):
        """ Construct an Attention Residual Block
            x         : input to the block
            strides   : strides
            n_filters : number of filters
            alpha     : width multiplier
            expansion : multiplier for expanding number of filters
            squeeze   : whether to include squeeze
            activation: type of activation function
        """
        n_filters = metaparameters['n_filters']
        expansion = metaparameters['expansion']
        alpha     = metaparameters['alpha']
        if 'alpha' in metaparameters:
            alpha = metaparameters['alpha']
        else:
            alpha = self.alpha
        if 'squeeze' in metaparameters:
            squeeze = metaparameters['squeeze']
        else:
            squeeze = False
        if 'activation' in metaparameters:
            activation = metaparameters['activation']
            del metaparameters['activation']
        else:
            activation = ReLU6
        del metaparameters['n_filters']
        del metaparameters['kernel_size']
            
        # Remember input
        shortcut = x

        # Apply the width filter to the number of feature maps for the pointwise convolution
        filters = int(n_filters * alpha)
    
        n_channels = int(x.shape[3])
    
        # Dimensionality Expansion
        # 1x1 linear convolution
        x = self.Conv2D(x, expansion, (1, 1), padding='same', **metaparameters)
        x = self.BatchNormalization(x)
        x = activation(x)

        # Depthwise Convolution
        x = self.DepthwiseConv2D(x, (3, 3), strides, padding='same', **metaparameters)
        x = self.BatchNormalization(x)
        x = activation(x)

        # Add squeeze (dimensionality reduction)
        if squeeze:
            x = self.squeeze(x, **metaparameters)

        # Linear Pointwise Convolution
        x = self.Conv2D(x, filters, (1, 1), strides=(1, 1), padding='same', **metaparameters)
        x = self.BatchNormalization(x)
    
        # Number of input filters matches the number of output filters
        if n_channels == filters and strides == (1, 1):
            x = Add()([shortcut, x]) 
        return x

    def squeeze(self, x, **metaparameters):
        """ Construct a squeeze block
            x   : input to the squeeze
        """
        if 'activation' in metaparameters:
            del metaparameters['activation']
        
        shortcut = x
        n_channels = x.shape[-1]

        x = GlobalAveragePooling2D()(x)
        x = self.Dense(x, n_channels, activation=self.ReLU, use_bias=False, **metaparameters)
        x = self.Dense(x, n_channels, activation=self.HS, use_bias=False, **metaparameters)
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

        x = self.Conv2D(x, 1280, (1, 1), padding='same', activation=self.HS, use_bias=True)

        # final classification
        x = self.Conv2D(x, n_classes, (1, 1), padding='same', activation='softmax', use_bias=True)
        # Flatten the feature maps into 1D feature maps (?, N)
        outputs = Reshape((n_classes,))(x)

        # Save post-activation layer
        self.softmax = outputs

        return outputs

# Example
# mobilenet = MobileNetV3('large')

def example():
    ''' Example for constructing/training a MobileNet V3 model on CIFAR-10
    '''
    # Example of constructing a mini-MobileNet
    mobilenet = MobileNetV3('small', input_shape=(32, 32, 3), n_classes=10)
    mobilenet.model.summary()
    mobilenet.cifar10()

# example()
