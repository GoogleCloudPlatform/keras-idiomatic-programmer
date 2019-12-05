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

# ResNet (50, 101, 152 + composable) version 2
# Paper: https://arxiv.org/pdf/1603.05027.pdf
# In this version, the BatchNormalization and ReLU activation are moved to be before the convolution in the bottleneck/projection blocks.
# In v1 and v1.5 they were after. 
# Note, this means that the ReLU that appeared after the add operation is now replaced as the ReLU proceeding the ending 1x1
# convolution in the block.

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, BatchNormalization, ReLU
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Add, Activation
from tensorflow.keras.regularizers import l2

import sys
sys.path.append('../')
from models_c import Composable

class ResNetV2(Composable):
    """ Construct a Residual Convolution Network Network V2 """
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
    _model = None

    def __init__(self, n_layers, input_shape=(224, 224, 3), n_classes=1000, reg=l2(0.001), relu=None, init_weights='he_normal'):
        """ Construct a Residual Convolutional Neural Network V2
            n_layers    : number of layers
            input_shape : input shape
            n_classes   : number of output classes
            reg         : kernel regularizer
            init_weights: kernel initializer
            relu        : max value for ReLU
        """
        # Configure base (super) class
        super().__init__(reg=reg, init_weights=init_weights, relu=relu)

        # predefined
        if isinstance(n_layers, int):
            if n_layers not in [50, 101, 152]:
                raise Exception("ResNet: Invalid value for n_layers")
            groups = self.groups[n_layers]
        # user defined
        else:
            groups = n_layers

        # The input tensor
        inputs = Input(input_shape)

        # The stem convolutional group
        x = self.stem(inputs, reg=reg)

        # The learner
        x = self.learner(x, groups=groups, reg=reg)

        # The classifier 
        outputs = self.classifier(x, n_classes, reg=reg)

        # Instantiate the Model
        self._model = Model(inputs, outputs)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, _model):
        self._model = _model

    def stem(self, inputs, **metaparameters):
        """ Construct the Stem Convolutional Group 
            inputs : the input vector
            reg    : kernel regularizer
        """
        reg = metaparameters['reg']

        # The 224x224 images are zero padded (black - no signal) to be 230x230 images prior to the first convolution
        x = ZeroPadding2D(padding=(3, 3))(inputs)
    
        # First Convolutional layer uses large (coarse) filter
        x = Conv2D(64, (7, 7), strides=(2, 2), padding='valid', use_bias=False, 
                   kernel_initializer=self.init_weights, kernel_regularizer=reg)(x)
        x = BatchNormalization()(x)
        x = Composable.ReLU(x)
    
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
        x = ResNetV2.group(x, strides=(1, 1), **groups.pop(0), **metaparameters)

        # Remaining Residual Block Groups (strided)
        for group in groups:
            x = ResNetV2.group(x, **group, **metaparameters)
        return x
    
    @staticmethod
    def group(x, strides=(2, 2), init_weights=None, **metaparameters):
        """ Construct a Residual Group
            x         : input into the group
            strides   : whether the projection block is a strided convolution
            n_blocks  : number of residual blocks with identity link
        """
        n_blocks  = metaparameters['n_blocks']

        # Double the size of filters to fit the first Residual Block
        x = ResNetV2.projection_block(x, strides=strides, init_weights=init_weights, **metaparameters)

        # Identity residual blocks
        for _ in range(n_blocks):
            x = ResNetV2.identity_block(x, init_weights=init_weights, **metaparameters)
        return x

    @staticmethod
    def identity_block(x, init_weights=None, **metaparameters):
        """ Construct a Bottleneck Residual Block with Identity Link
            x        : input into the block
            n_filters: number of filters
            reg      : kernel regularizer
        """
        n_filters = metaparameters['n_filters']
        if 'reg' in metaparameters:
            reg = metaparameters['reg']
        else:
            reg = ResNetV2.reg

        if init_weights is None:
            init_weights = ResNetV2.init_weights
    
        # Save input vector (feature maps) for the identity link
        shortcut = x
    
        ## Construct the 1x1, 3x3, 1x1 convolution block
    
        # Dimensionality reduction
        x = BatchNormalization()(x)
        x = Composable.ReLU(x)
        x = Conv2D(n_filters, (1, 1), strides=(1, 1), use_bias=False, 
                   kernel_initializer=init_weights, kernel_regularizer=reg)(x)

        # Bottleneck layer
        x = BatchNormalization()(x)
        x = Composable.ReLU(x)
        x = Conv2D(n_filters, (3, 3), strides=(1, 1), padding="same", use_bias=False, 
                   kernel_initializer=init_weights, kernel_regularizer=reg)(x)

        # Dimensionality restoration - increase the number of output filters by 4X
        x = BatchNormalization()(x)
        x = Composable.ReLU(x)
        x = Conv2D(n_filters * 4, (1, 1), strides=(1, 1), use_bias=False, 
                   kernel_initializer=init_weights, kernel_regularizer=reg)(x)

        # Add the identity link (input) to the output of the residual block
        x = Add()([shortcut, x])
        return x

    @staticmethod
    def projection_block(x, strides=(2,2), init_weights=None, **metaparameters):
        """ Construct a Bottleneck Residual Block of Convolutions with Projection Shortcut
            Increase the number of filters by 4X
            x        : input into the block
            strides  : whether the first convolution is strided
            n_filters: number of filters
            reg      : kernel regularizer
        """
        n_filters = metaparameters['n_filters']
        if 'reg' in metaparameters:
            reg = metaparameters['reg']
        else:
            reg = ResNetV2.reg

        if init_weights is None:
            init_weights = ResNetV2.init_weights

        # Construct the projection shortcut
        # Increase filters by 4X to match shape when added to output of block
        shortcut = BatchNormalization()(x)
        shortcut = Conv2D(4 * n_filters, (1, 1), strides=strides, use_bias=False, 
                          kernel_initializer=init_weights, kernel_regularizer=reg)(shortcut)

        ## Construct the 1x1, 3x3, 1x1 convolution block
    
        # Dimensionality reduction
        x = BatchNormalization()(x)
        x = Composable.ReLU(x)
        x = Conv2D(n_filters, (1, 1), strides=(1,1), use_bias=False, 
                   kernel_initializer=init_weights, kernel_regularizer=reg)(x)

        # Bottleneck layer
        # Feature pooling when strides=(2, 2)
        x = BatchNormalization()(x)
        x = Composable.ReLU(x)
        x = Conv2D(n_filters, (3, 3), strides=strides, padding='same', use_bias=False, 
                   kernel_initializer=init_weights, kernel_regularizer=reg)(x)

        # Dimensionality restoration - increase the number of filters by 4X
        x = BatchNormalization()(x)
        x = Composable.ReLU(x)
        x = Conv2D(4 * n_filters, (1, 1), strides=(1, 1), use_bias=False, 
                   kernel_initializer=init_weights, kernel_regularizer=reg)(x)

        # Add the projection shortcut to the output of the residual block
        x = Add()([x, shortcut])
        return x

    def classifier(self, x, n_classes, **metaparameters):
        """ Construct the Classifier Group 
            x         : input to the classifier
            n_classes : number of output classes
        """
        reg = metaparameters['reg']

        # Save the encoding layer
        self.encoding = x
        
        # Pool at the end of all the convolutional residual blocks
        x = GlobalAveragePooling2D()(x)

        # Save the bottleneck layer
        self.bottleneck = x

        # Final Dense Outputting Layer for the outputs
        x = Dense(n_classes, 
                        kernel_initializer=self.init_weights, kernel_regularizer=reg)(x)
        # Save the pre-activation probabilities layer
        self.probabilities = x
        outputs = Activation('softmax')(x)
        # Save the post-activation probabilities layer
        self.softmax = outputs
        return outputs

# Example
# resnet = ResNetV2(50)
