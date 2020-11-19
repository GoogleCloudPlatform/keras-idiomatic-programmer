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

# ResNetb (50, 101, 152 + composable) version 1.5
# Trainable params: 31,487,272
# Paper: https://arxiv.org/pdf/1512.03385.pdf
# The strides=2 in the projection_block is moved from the 1x1 convolution to the 
# 3x3 convolution. Gained 0.5% more accuracy on ImageNet

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, BatchNormalization
from tensorflow.keras.layers import ReLU, Dense, GlobalAveragePooling2D, Add, Activation
from tensorflow.keras.regularizers import l2

import sys
sys.path.append('../')
from models_c import Composable

class ResNetV1_5(Composable):
    """ Construct a Residual Convoluntional Network V1.5 """
    # Meta-parameter: list of groups: number of filters and number of blocks
    groups = { 50 : [ { 'n_filters' : 64, 'n_blocks': 3 },
                      { 'n_filters': 128, 'n_blocks': 4 },
                      { 'n_filters': 256, 'n_blocks': 6 },
                      { 'n_filters': 512, 'n_blocks': 3 } ],            # ResNet50
               101: [ { 'n_filters' : 64, 'n_blocks': 3 },
                      { 'n_filters': 128, 'n_blocks': 4 },
                      { 'n_filters': 256, 'n_blocks': 23 },
                      { 'n_filters': 512, 'n_blocks': 3 } ],            # ResNet101
               152: [ { 'n_filters' : 64, 'n_blocks': 3 },
                      { 'n_filters': 128, 'n_blocks': 8 },
                      { 'n_filters': 256, 'n_blocks': 36 },
                      { 'n_filters': 512, 'n_blocks': 3 } ]             # ResNet152
             }

    # Initial Hyperparameters
    hyperparameters = { 'initializer': 'he_normal',
                        'regularizer': l2(0.001),
                        'relu_clip'  : None,
                        'bn_epsilon' : None,
                        'use_bias'   : False
                      }
    
    def __init__(self, n_layers, 
                 input_shape=(224, 224, 3), n_classes=1000, include_top=True,
                 **hyperparameters):
        """ Construct a Residual Convolutional Neural Network V1.5
            n_layers    : number of layers
            input_shape : input shape
            n_classes   : number of output classes
            include_top : whether to include classifier
            regularizer : kernel regularizer
            relu_clip   : max value for ReLU
            initializer : kernel initializer
            bn_epsilon  : epsilon on batch norm
            use_bias    : whether to use bias with batchnorm
        """
        # Configure base (super) class
        Composable.__init__(self, input_shape, include_top, self.hyperparameters, **hyperparameters)

        # pr edefined
        if isinstance(n_layers, int):
            if n_layers not in [50, 101, 152]:
                raise Exception("ResNet: Invalid value for n_layers")
            groups = list(self.groups[n_layers])
        # user defined
        else:
            groups = n_layers

        # The input tensor
        inputs = Input(input_shape)

        # The stem convolutional group
        x = self.stem(inputs)

        # The learner
        outputs = self.learner(x, groups=groups)

        # The classifier 
        if include_top:
            # Add hidden dropout for training-time regularization
            outputs = self.classifier(outputs, n_classes, dropout=0.0)

        # Instantiate the Model
        self._model = Model(inputs, outputs)
    
    def stem(self, inputs):
        """ Construct the Stem Convolutional Group 
            inputs : the input vector
        """
        # The 224x224 images are zero padded (black - no signal) to be 230x230 images prior to the first convolution
        x = ZeroPadding2D(padding=(3, 3))(inputs)
    
        # First Convolutional layer uses large (coarse) filter
        x = self.Conv2D(x, 64, (7, 7), strides=(2, 2), padding='valid')
        x = self.BatchNormalization(x)
        x = self.ReLU(x)
    
        # Pooled feature maps will be reduced by 75%
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)
        return x

    def learner(self, x, **metaparameters):
        """ Construct the Learner
            x     : input to the learner
            groups: list of groups: number of filters and blocks
        """
        groups = metaparameters['groups']

        # First Residual Block Group (not strided)
        x = self.group(x, strides=(1, 1), **groups.pop(0))

        # Remaining Residual Block Groups (strided)
        for group in groups:
            x = self.group(x, **group)
        return x

    def group(self, x, strides=(2, 2), **metaparameters):
        """ Construct a Residual Group
            x         : input into the group
            strides   : whether the projection block is a strided convolution
            n_blocks  : number of residual blocks with identity link
        """
        n_blocks  = metaparameters['n_blocks']

        # Double the size of filters to fit the first Residual Block
        x = self.projection_block(x, strides=strides, **metaparameters)

        # Identity residual blocks
        for _ in range(n_blocks):
            x = self.identity_block(x, **metaparameters)
        return x

    def identity_block(self, x, **metaparameters):
        """ Construct a Bottleneck Residual Block with Identity Link
            x        : input into the block
            n_filters: number of filters
        """
        n_filters = metaparameters['n_filters']
        del metaparameters['n_filters']
    
        # Save input vector (feature maps) for the identity link
        shortcut = x
    
        ## Construct the 1x1, 3x3, 1x1 residual block
    
        # Dimensionality reduction
        x = self.Conv2D(x, n_filters, (1, 1), strides=(1, 1), **metaparameters)
        x = self.BatchNormalization(x)
        x = self.ReLU(x)

        # Bottleneck layer
        x = self.Conv2D(x, n_filters, (3, 3), strides=(1, 1), padding="same", **metaparameters)
        x = self.BatchNormalization(x)
        x = self.ReLU(x)

        # Dimensionality restoration - increase the number of output filters by 4X
        x = self.Conv2D(x, n_filters * 4, (1, 1), strides=(1, 1), **metaparameters)
        x = self.BatchNormalization(x)

        # Add the identity link (input) to the output of the residual block
        x = Add()([shortcut, x])
        x = self.ReLU(x)
        return x

    def projection_block(self, x, strides=(2, 2), **metaparameters):
        """ Construct Bottleneck Residual Block of Convolutions with Projection Shortcut
            Increase the number of filters by 4X
            x        : input into the block
            strides  : whether the first convolution is strided
            n_filters: number of filters
        """
        n_filters = metaparameters['n_filters']
        del metaparameters['n_filters']
    
        # Construct the projection shortcut
        # Increase filters by 4X to match shape when added to output of block
        shortcut = self.Conv2D(x, 4 * n_filters, (1, 1), strides=strides, **metaparameters)
        shortcut = self.BatchNormalization(shortcut)

        ## Construct the 1x1, 3x3, 1x1 residual block

        # Dimensionality reduction
        x = self.Conv2D(x, n_filters, (1, 1), strides=(1, 1), **metaparameters)
        x = self.BatchNormalization(x)
        x = self.ReLU(x)

        # Bottleneck layer
        # Feature pooling when strides=(2, 2)
        x = self.Conv2D(x, n_filters, (3, 3), strides=strides, padding='same', **metaparameters)
        x = self.BatchNormalization(x)
        x = self.ReLU(x)

        # Dimensionality restoration - increase the number of output filters by 4X
        x = self.Conv2D(x, 4 * n_filters, (1, 1), strides=(1, 1), **metaparameters)
        x = self.BatchNormalization(x)

        # Add the projection shortcut to the output of the residual block
        x = Add()([x, shortcut])
        x = self.ReLU(x)
        return x


# Example of a ResNet50 V1.5
# resnet = ResNetV1_5(50)

def example():
    ''' Example for constructing/training a ResNet V1.5 model on CIFAR-10
    '''
    # Example of constructing a mini-ResNet
    groups = [ { 'n_filters' : 64, 'n_blocks': 1 },
               { 'n_filters': 128, 'n_blocks': 2 },
               { 'n_filters': 256, 'n_blocks': 2 }]
    resnet = ResNetV1_5(groups, input_shape=(32, 32, 3), n_classes=10)
    resnet.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
    resnet.model.summary()
    resnet.cifar10()

example()
