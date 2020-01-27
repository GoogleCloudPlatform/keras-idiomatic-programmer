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

# ResNet20, 56, 110, 164, 1001 version 1 for CIFAR-10
# Paper: https://arxiv.org/pdf/1512.03385.pdf

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Add, Dense
from tensorflow.keras.layers import AveragePooling2D, Flatten
from tensorflow.keras.regularizers import l2

import sys
sys.path.append('../')
from models_c import Composable

class ResNetCifarV1(Composable):
    """ Residual Convolutional Neural Network V1
    """
    #-------------------
    # Model      | n   |
    # ResNet20   | 2   |
    # ResNet56   | 6   |
    # ResNet110  | 12  |
    # ResNet164  | 18  |
    # ResNet1001 | 111 |

    def __init__(self, n=2, m=2,
                 input_shape=(32, 32, 3), n_classes=10, include_top=True,
                 reg=l2(0.001), relu=None, init_weights='he_normal', bias=False):
        """ Construct a Residual Convolutional Neural Network V1
            n           :
            input_shape : input shape
            n_classes   : number of output classes
            include_top : whether to include classifier
            reg         : kernel regularizer
            relu        : max value for ReLU
            init_weights: kernel initializer
            bias        : whether to use bias with batchnorm
        """
        # Configure the base (super) class
        super().__init__(reg=reg, relu=relu, init_weights=init_weights, bias=bias)

        if n not in [2, 6, 12, 18, 111]:
            raise ""

        depth =  n * 9 + 2
        n_blocks = ((depth - 2) // 9) - 1

        # The input tensor
        inputs = Input(input_shape)

        # The stem convolutional group
        x = self.stem(inputs)

        # The learner
        outputs = self.learner(x, n_blocks=n_blocks)

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
            x          : input to the learner
            n_blocks   : number of blocks in a group
        """
        n_blocks = metaparameters['n_blocks']

        # First Residual Block Group of 16 filters (Stage 1)
        # Quadruple (4X) the size of filters to fit the next Residual Group
        x = self.group(x, 16, n_blocks, strides=(1, 1), m=4)

        # Second Residual Block Group of 64 filters (Stage 2)
        # Double (2X) the size of filters and reduce feature maps by 75% (strides=2) to fit the next Residual Group
        x = self.group(x, 64, n_blocks, m=2)

        # Third Residual Block Group of 64 filters (Stage 3)
        # Double (2X) the size of filters and reduce feature maps by 75% (strides=2) to fit the next Residual Group
        x = self.group(x, 128, n_blocks, m=2)
        return x

    def group(self, x, n_filters, n_blocks, strides=(2, 2), m=2):
        """ Construct a Residual Group
            x         : input into the group
            n_filters : number of filters for the group
            n_blocks  : number of residual blocks with identity link
            strides   : whether the projection block is a strided convolution
            m         : multiplier for the number of filters out
        """
        # Double the size of filters to fit the first Residual Group
        x = self.projection_block(x, n_filters, strides=strides, m=m)

        # Identity residual blocks
        for _ in range(n_blocks):
            x = self.identity_block(x, n_filters, m=m)
        return x
    
    def identity_block(self, x, n_filters, m=2):
        """ Construct a Bottleneck Residual Block of Convolutions with Identity Shortcut
            x        : input into the block
            n_filters: number of filters
            m        : multiplier for filters out
        """
        # Save input vector (feature maps) for the identity link
        shortcut = x
    
        ## Construct the 1x1, 3x3, 1x1 residual block (fig 3c)

        # Dimensionality reduction
        x = self.Conv2D(x, n_filters, (1, 1), strides=(1, 1))
        x = self.BatchNormalization(x)
        x = self.ReLU(x)

        # Bottleneck layer
        x = self.Conv2D(x, n_filters, (3, 3), strides=(1, 1), padding="same")
        x = self.BatchNormalization(x)
        x = self.ReLU(x)

        # Dimensionality restoration - increase the number of output filters by 2X or 4X
        x = self.Conv2D(x, n_filters * m, (1, 1), strides=(1, 1))
        x = self.BatchNormalization(x)

        # Add the identity link (input) to the output of the residual block
        x = Add()([x, shortcut])
        x = self.ReLU(x)
        return x

    def projection_block(self, x, n_filters, strides=(2,2), m=2):
        """ Construct Bottleneck Residual Block with Projection Shortcut
            Increase the number of filters by 2X (or 4X on first stage)
            x        : input into the block
            n_filters: number of filters
            strides  : whether the first convolution is strided
            m        : multiplier for number of filters out
        """
        # Construct the projection shortcut
        # Increase filters by 2X (or 4X) to match shape when added to output of block
        shortcut = self.Conv2D(x, n_filters * m, (1, 1), strides=strides)

        ## Construct the 1x1, 3x3, 1x1 convolution block

        # Dimensionality reduction
        x = self.Conv2D(x, n_filters, (1, 1), strides=(1,1))
        x = self.BatchNormalization(x)
        x = self.ReLU(x)
    
        # Bottleneck layer - feature pooling when strides=(2, 2)
        x = self.Conv2D(x, n_filters, (3, 3), strides=strides, padding='same')
        x = self.BatchNormalization(x)
        x = self.ReLU(x)  
    
        # Dimensionality restoration - increase the number of filters by 2X (or 4X)
        x = self.Conv2D(x, n_filters * m, (1, 1), strides=(1, 1))
        x = self.BatchNormalization(x)

        # Add the projection shortcut to the output of the residual block
        x = Add()([shortcut, x])
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
        x = Flatten()(x)

        # Final Dense Outputting Layer 
        outputs = self.Dense(x, n_classes, activation='softmax')
        return outputs

# Example
cifar = ResNetCifarV1(2)
# wcifar.model.summary()
