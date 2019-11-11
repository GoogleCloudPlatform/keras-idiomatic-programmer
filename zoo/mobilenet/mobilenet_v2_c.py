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


# MobileNet v2 + composable (2019)
# Paper: https://arxiv.org/pdf/1801.04381.pdf
# 224x224 input: 3,504,872 parameters

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import ZeroPadding2D, Conv2D, BatchNormalization, ReLU
from tensorflow.keras.layers import DepthwiseConv2D, Add, GlobalAveragePooling2D, Dense
from tensorflow.keras.regularizers import l2

class MobileNetV2(object):
    """ Construct a Mobile Convolution Neural Network """
    # Meta-parameter: number of filters and blocks per group
    groups = [ { 'n_filters' : 16,   'n_blocks' : 1 }, 
               { 'n_filters' : 24,   'n_blocks' : 2 },
               { 'n_filters' : 32,   'n_blocks' : 3 }, 
               { 'n_filters' : 64,   'n_blocks' : 4 },
               { 'n_filters' : 96,   'n_blocks' : 3 },
               { 'n_filters' : 160,  'n_blocks' : 3 }, 
               { 'n_filters' : 320,  'n_blocks' : 1 },
               { 'n_filters' : 1280, 'n_blocks' : 1 } ]

    # Meta-parameter: width multiplier (0 .. 1) for reducing number of filters.
    alpha = 1
    # Meta-parameter: multiplier to expand the number of filters
    expansion = 6
    # Meta-parameter: kernel regularization
    reg = l2(0.001)
    init_weights = 'glorot_uniform'
    _model = None

    def __init__(self, groups=None, alpha=1, expansion=6, input_shape=(224, 224, 3), n_classes=1000):
        """ Construct a Mobile Convolution Neural Network
            groups     : number of filters and blocks per group
            alpha      : width multiplier
            expansion  : multiplier to expand the number of filters
            input_shape: the input shape
            n_classes  : number of output classes
        """
        if groups is None:
             groups = self.groups

        inputs = Input(shape=(224, 224, 3))

        # The Stem Group
        x = self.stem(inputs, alpha=alpha)    

        # The Learner
        x = self.learner(x, groups=groups, alpha=alpha, expansion=expansion)

        # The Classifier 
        outputs = self.classifier(x, n_classes)

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
        """
        alpha = metaparameters['alpha']

        # Calculate the number of filters for the stem convolution
        # Must be divisible by 8
        n_filters = max(8, (int(32 * alpha) + 4) // 8 * 8)
    
        # Convolutional block
        x = ZeroPadding2D(padding=((0, 1), (0, 1)))(inputs)
        x = Conv2D(n_filters, (3, 3), strides=(2, 2), padding='valid', use_bias=False, 
                   kernel_initializer=self.init_weights, kernel_regularizer=self.reg)(x)
        x = BatchNormalization()(x)
        x = ReLU(6.)(x)
        return x
    
    def learner(self, x, **metaparameters):
        """ Construct the Learner
            x        : input to the learner
            alpha    : width multiplier
            expansion: multipler to expand number of filters
        """
        groups = metaparameters['groups']
        alpha  = metaparameters['alpha']
        expansion = metaparameters['expansion']

        last = groups.pop()

        # First Inverted Residual Convolution Group
        group = groups.pop(0)
        x = MobileNetV2.group(x, **group, alpha=alpha, expansion=1, strides=(1, 1))

        # Add Inverted Residual Convolution Group
        for group in groups:
            x = MobileNetV2.group(x, **group, alpha=alpha, expansion=expansion)

        # Last block is a 1x1 linear convolutional layer,
        # expanding the number of filters to 1280.
        x = Conv2D(1280, (1, 1), use_bias=False, kernel_initializer=self.init_weights)(x)
        x = BatchNormalization()(x)
        x = ReLU(6.)(x)
        return x

    @staticmethod
    def group(x, strides=(2, 2), **metaparameters):
        """ Construct an Inverted Residual Group
            x         : input to the group
            strides   : whether first inverted residual block is strided.
            n_blocks  : number of blocks in the group
        """   
        n_blocks  = metaparameters['n_blocks']

        # In first block, the inverted residual block maybe strided - feature map size reduction
        x = MobileNetV2.inverted_block(x, strides=strides, **metaparameters)
    
        # Remaining blocks
        for _ in range(n_blocks - 1):
            x = MobileNetV2.inverted_block(x, strides=(1, 1), **metaparameters)
        return x

    @staticmethod
    def inverted_block(x, strides=(1, 1), init_weights=None, **metaparameters):
        """ Construct an Inverted Residual Block
            x         : input to the block
            strides   : strides
            n_filters : number of filters
            alpha     : width multiplier
            expansion : multiplier for expanding number of filters
            reg       : kernel regularizer
        """
        n_filters = metaparameters['n_filters']
        alpha     = metaparameters['alpha']
        if 'alpha' in metaparameters:
            alpha = metaparameters['alpha']
        else:
            alpha = MobileNetV2.alpha
        if 'expansion' in metaparameters:
            expansion = metaparameters['expansion']
        else:
            expansion = MobileNetV2.expansion
        if 'reg' in metaparameters:
            reg = metaparameters['reg']
        else:
            reg = MobileNetV2.reg

        if init_weights is None:
            init_weights = MobileNetV2.init_weights
            
        # Remember input
        shortcut = x

        # Apply the width filter to the number of feature maps for the pointwise convolution
        filters = int(n_filters * alpha)
    
        n_channels = int(x.shape[3])
    
        # Dimensionality Expansion (non-first block)
        if expansion > 1:
            # 1x1 linear convolution
            x = Conv2D(expansion * n_channels, (1, 1), padding='same', use_bias=False, 
                       kernel_initializer=init_weights, kernel_regularizer=reg)(x)
            x = BatchNormalization()(x)
            x = ReLU(6.)(x)

        # Strided convolution to match number of filters
        if strides == (2, 2):
            x = ZeroPadding2D(padding=((0, 1), (0, 1)))(x)
            padding = 'valid'
        else:
            padding = 'same'

        # Depthwise Convolution
        x = DepthwiseConv2D((3, 3), strides, padding=padding, use_bias=False, kernel_initializer=init_weights, 
                            kernel_regularizer=reg)(x)
        x = BatchNormalization()(x)
        x = ReLU(6.)(x)

        # Linear Pointwise Convolution
        x = Conv2D(filters, (1, 1), strides=(1, 1), padding='same', use_bias=False, 
                   kernel_initializer=init_weights, kernel_regularizer=reg)(x)
        x = BatchNormalization()(x)
    
        # Number of input filters matches the number of output filters
        if n_channels == filters and strides == (1, 1):
            x = Add()([shortcut, x]) 
        return x

    def classifier(self, x, n_classes):
        """ Construct the classifier group
            x         : input to the classifier
            n_classes : number of output classes
        """
        # Flatten the feature maps into 1D feature maps (?, N)
        x = GlobalAveragePooling2D()(x)

        # Dense layer for final classification
        x = Dense(n_classes, activation='softmax', kernel_initializer=self.init_weights, kernel_regularizer=self.reg)(x)
        return x

# Example
# mobilenet = MobileNetV2()
