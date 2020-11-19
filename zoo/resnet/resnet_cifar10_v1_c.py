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

# ResNet20, 32, 44, 56, 110 version 1 for CIFAR-10
# Paper: https://arxiv.org/pdf/1512.03385.pdf

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Add, Dense
from tensorflow.keras.layers import AveragePooling2D, Flatten, Dropout, Activation
from tensorflow.keras.regularizers import l2

import sys
sys.path.append('../')
from models_c import Composable

class ResNetCifarV1(Composable):
    """ Residual Convolutional Neural Network V1 for CIFAR-10
    """
    groups = { 20 : [ { 'n_filters': 16, 'n_blocks': 3},
                      { 'n_filters': 32, 'n_blocks': 3},
                      { 'n_filters': 64, 'n_blocks': 3}],
               32 : [ { 'n_filters': 16, 'n_blocks': 5},
                      { 'n_filters': 32, 'n_blocks': 5},
                      { 'n_filters': 64, 'n_blocks': 5}],
               44 : [ { 'n_filters': 16, 'n_blocks': 7},
                      { 'n_filters': 32, 'n_blocks': 7},
                      { 'n_filters': 64, 'n_blocks': 7}],
               56 : [ { 'n_filters': 16, 'n_blocks': 9},
                      { 'n_filters': 32, 'n_blocks': 9},
                      { 'n_filters': 64, 'n_blocks': 9}],
               110: [ { 'n_filters': 16, 'n_blocks': 18},
                      { 'n_filters': 32, 'n_blocks': 18},
                      { 'n_filters': 64, 'n_blocks': 18}],
               164: [ { 'n_filters': 16, 'n_blocks': 27},
                      { 'n_filters': 32, 'n_blocks': 27},
                      { 'n_filters': 64, 'n_blocks': 27}]}

    # Initial Hyperparameters
    hyperparameters = { 'initializer': 'he_normal',
                        'regularizer': l2(0.001),
                        'relu_clip'  : None,
                        'bn_epsilon' : None,
                        'use_bias'   : False
                      }

    def __init__(self, n_layers,
                 input_shape=(32, 32, 3), n_classes=10, include_top=True,
                 **hyperparameters):
        """ Construct a Residual Convolutional Neural Network V1
            n_layers    : number of layers
            input_shape : input shape
            n_classes   : number of output classes
            include_top : whether to include classifier
            regularizer : kernel regularizer
            relu_clip   : max value for ReLU
            initializer : kernel initializer
            bn_epsilon  : epsilon for batch norm
            use_bias    : whether to use bias with batchnorm
        """
        # Configure the base (super) class
        Composable.__init__(self, input_shape, include_top, self.hyperparameters, **hyperparameters)

        # depth
        if isinstance(n_layers, int):
            if n_layers not in [20, 32, 44, 56, 110, 164]:
                raise Exception("ResNet CIFAR: invalid value for n_layers")
            groups = list(self.groups[n_layers])
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
            outputs = self.classifier(outputs, n_classes)

        # Instantiate the Model
        self._model = Model(inputs, outputs)

    def stem(self, inputs):
        ''' Construct the Stem Convolutional Group 
            inputs : the input vector
        '''
        x = self.Conv2D(inputs, 16, (3, 3), strides=(1, 1), padding='same')
        x = self.BatchNormalization(x)
        x = self.ReLU(x)
        return x
    
    def learner(self, x, **metaparameters):
        """ Construct the Learner
            x        : input to the learner
            groups   : filter/blocks per group
        """
        groups = metaparameters['groups']

        # first group uses strides=1 for first convolution
        x = self.group(x, **groups.pop(0), proj=False)

        # remaining greoups
        for group in groups:
            x = self.group(x, **group)

        return x

    def group(self, x, n_filters, n_blocks, proj=True):
        """ Construct a Residual Group
            x         : input into the group
            n_filters : number of filters for the group
            n_blocks  : number of residual blocks with identity link
            proj      : whether first block is a projection block
        """
        if proj:
            x = self.projection_block(x, n_filters)
        else:
            x = self.identity_block(x, n_filters)

        # Identity residual blocks
        for _ in range(n_blocks-1):
            x = self.identity_block(x, n_filters)
        return x
    
    def identity_block(self, x, n_filters):
        """ Construct a Bottleneck Residual Block of Convolutions with Identity Shortcut
            x        : input into the block
            n_filters: number of filters
        """
        # Save input vector (feature maps) for the identity link
        shortcut = x

        # Two 3x3 post-activation convolutuional layers
    
        x = self.Conv2D(x, n_filters, (3, 3), strides=(1, 1), padding='same')
        x = self.BatchNormalization(x)
        x = self.ReLU(x)

        x = self.Conv2D(x, n_filters, (3, 3), strides=(1, 1), padding='same')
        x = self.BatchNormalization(x)

        # Add the identity link (input) to the output of the residual block
        x = Add()([x, shortcut])
        x = self.ReLU(x)
        return x
    
    def projection_block(self, x, n_filters):
        """ Construct a Bottleneck Residual Block of Convolutions with Identity Shortcut
            x        : input into the block
            n_filters: number of filters
        """
        # Save input vector (feature maps) for the identity link
        shortcut = x
        shortcut = self.Conv2D(shortcut, n_filters, (1, 1), strides=(2, 2), padding='same')
    
        x = self.Conv2D(x, n_filters, (3, 3), strides=(2, 2), padding='same')
        x = self.BatchNormalization(x)
        x = self.ReLU(x)

        x = self.Conv2D(x, n_filters, (3, 3), strides=(1, 1), padding='same')
        x = self.BatchNormalization(x)

        # Add the identity link (input) to the output of the residual block
        x = Add()([x, shortcut])
        x = self.ReLU(x)
        return x

    def classifier(self, x, n_classes):
        ''' Construct the Classifier
            x         : input into the classifier
            n_classes : number of classes
        '''
        # Pool the feature maps after the end of all the residual blocks
        x = self.BatchNormalization(x)
        x = self.ReLU(x)
        x = AveragePooling2D(pool_size=8)(x)
    
        # Flatten into 1D vector
        self.encoding = x
        x = Flatten()(x)
        self.embedding = x

        # Add hidden dropout unit
        x = Dropout(0.0)(x)

        # Final Dense Outputting Layer 
        outputs = self.Dense(x, n_classes)
        self.probabilities = outputs
        outputs = Activation('softmax')(outputs)
        self.softmax = outputs
        return outputs

# Example
# cifar = ResNetCifarV1(20)

def example():
    ''' Example for constructing/training a ResNet V2 model on CIFAR-10
    '''
    # Example of constructing a mini-ResNet
    resnet = ResNetCifarV1(20, input_shape=(32, 32, 3), n_classes=10)
    resnet.model.summary()
    resnet.cifar10(decay=('time', 1e-4))

# example()
