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


# MobileNet 224 + Composable (2017)
# Note: 224 refers to the image size, not the number of layers
# Paper: https://arxiv.org/pdf/1704.04861.pdf

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import ZeroPadding2D, Conv2D, BatchNormalization, ReLU, Activation
from tensorflow.keras.layers import DepthwiseConv2D, GlobalAveragePooling2D, Reshape, Dropout
from tensorflow.keras.regularizers import l2

import sys
sys.path.append('../')
from models_c import Composable

class MobileNetV1(Composable):
    """ Construct a Mobile Convolution Neural Network """
    # Meta-parameter: number of filters and number of blocks per group
    groups = [ { 'n_filters': 128,  'n_blocks': 2 },
               { 'n_filters': 256,  'n_blocks': 2 },
               { 'n_filters': 512,  'n_blocks': 6 },
               { 'n_filters': 1024, 'n_blocks': 2 } ]

    # Meta-parameter: width multiplier (0 .. 1) for reducing number of filters.
    alpha      = 1   
    # Meta-parameter: resolution multiplier (0 .. 1) for reducing input size
    pho        = 1
    # Meta-parameter: dropout rate
    dropout    = 0.5

    init_weights = 'glorot_uniform'
    relu = 6.0

    def __init__(self, alpha=1, pho=1, dropout=0.5, groups=None, input_shape=(224, 224, 3), n_classes=1000,
                 init_weights='glorot_uniform', reg=l2(0.001), relu=6.0):
        """ Construct a Mobile Convolution Neural Network
            alpha       : width multipler
            pho         : resolution multiplier
            input_shape : the input shape
            n_classes   : number of output classes
            init_weights: kernel initializer
            reg         : kernel regularizer
            relu        : max value for ReLU
        """
        # Configure base (super) class
        super().__init__(init_weights=init_weights, reg=reg, relu=6.0)
        
        if groups is None:
             groups = self.groups

        if alpha < 0 or alpha > 1:
            raise Exception("MobileNet: alpha out of range")
        if pho < 0 or pho > 1:
            raise Exception("MobileNet: pho out of range")
        if dropout < 0 or dropout > 1:
            raise Exception("MobileNet: alpha out of range")
        
        inputs = Input(shape=(int(input_shape[0] * pho), int(input_shape[1] * pho), 3))

        # The Stem Group
        x = self.stem(inputs, alpha=alpha)    

        # The Learner
        x = self.learner(x, groups=groups, alpha=alpha)

        # The Classifier 
        outputs = self.classifier(x, n_classes, alpha=alpha, dropout=dropout)

        # Instantiate the Model
        self._model = Model(inputs, outputs)
    
    def stem(self, inputs, **metaparameters):
        """ Construct the Stem Group
            inputs : input tensor
            alpha  : width multiplier
        """
        alpha = metaparameters['alpha']

        # Convolutional block
        x = ZeroPadding2D(padding=((0, 1), (0, 1)))(inputs)
        x = Conv2D(32 * alpha, (3, 3), strides=(2, 2), padding='valid', use_bias=False, 
                   kernel_initializer=self.init_weights, kernel_regularizer=self.reg)(x)
        x = BatchNormalization()(x)
        x = Composable.ReLU(x)

        # Depthwise Separable Convolution Block
        x = MobileNetV1.depthwise_block(x, (1, 1), n_filters=64, alpha=alpha)
        return x
    
    def learner(self, x, **metaparameters):
        """ Construct the Learner
            x      : input to the learner
            alpha  : width multiplier
            groups : number of filters and blocks per group
        """
        alpha  = metaparameters['alpha']
        groups = metaparameters['groups']

        # Add Depthwise Separable Convolution Group
        for group in groups:
            x = MobileNetV1.group(x, **group, alpha=alpha)

        return x

    @staticmethod
    def group(x, init_weights=None, **metaparameters):
        """ Construct a Depthwise Separable Convolution Group
            x         : input to the group
            n_blocks  : number of blocks in the group
        """   
        n_blocks  = metaparameters['n_blocks']

        # In first block, the depthwise convolution is strided - feature map size reduction
        x = MobileNetV1.depthwise_block(x, strides=(2, 2), init_weights=init_weights, **metaparameters)
    
        # Remaining blocks
        for _ in range(n_blocks - 1):
            x = MobileNetV1.depthwise_block(x, strides=(1, 1), init_weights=init_weights, **metaparameters)
        return x

    @staticmethod
    def depthwise_block(x, strides, init_weights=None, **metaparameters):
        """ Construct a Depthwise Separable Convolution block
            x         : input to the block
            strides   : strides
            n_filters : number of filters
            alpha     : width multiplier
            reg       : kernel regularizer
        """
        n_filters = metaparameters['n_filters']
        alpha     = metaparameters['alpha']
        if 'reg' in metaparameters:
            reg = metaparameters['reg']
        else:
            reg = MobileNetV1.reg

        if init_weights is None:
            init_weights = MobileNetV1.init_weights
            
        # Apply the width filter to the number of feature maps
        filters = int(n_filters * alpha)

        # Strided convolution to match number of filters
        if strides == (2, 2):
            x = ZeroPadding2D(padding=((0, 1), (0, 1)))(x)
            padding = 'valid'
        else:
            padding = 'same'

        # Depthwise Convolution
        x = DepthwiseConv2D((3, 3), strides, padding=padding, use_bias=False, 
                            kernel_initializer=init_weights, kernel_regularizer=reg)(x)
        x = BatchNormalization()(x)
        x = Composable.ReLU(x)

        # Pointwise Convolution
        x = Conv2D(filters, (1, 1), strides=(1, 1), padding='same', use_bias=False, 
                   kernel_initializer=init_weights, kernel_regularizer=reg)(x)
        x = BatchNormalization()(x)
        x = Composable.ReLU(x)
        return x

    def classifier(self, x, n_classes, **metaparameters):
        """ Construct the classifier group
            x         : input to the classifier
            alpha     : width multiplier
            dropout   : dropout percentage
            n_classes : number of output classes
        """
        alpha   = metaparameters['alpha']
        dropout = metaparameters['dropout']

        # Save encoding layer
        self.encoding = x

        # Flatten the feature maps into 1D feature maps (?, N)
        x = GlobalAveragePooling2D()(x)

        # Reshape the feature maps to (?, 1, 1, 1024)
        shape = (1, 1, int(1024 * alpha))
        x = Reshape(shape)(x)

        # Save embedding layer
        self.embedding = x
        
        # Perform dropout for preventing overfitting
        x = Dropout(dropout)(x)

        # Use convolution for classifying (emulates a fully connected layer)
        x = Conv2D(n_classes, (1, 1), padding='same', activation='softmax', 
                   kernel_initializer=self.init_weights, kernel_regularizer=self.reg)(x)
        # Reshape the resulting output to 1D vector of number of classes
        outputs = Reshape((n_classes, ))(x)

        # Save the post-activation probabilities layer
        self.softmax = outputs
        return outputs

# Example
# mobilenet = MobileNetV1()
