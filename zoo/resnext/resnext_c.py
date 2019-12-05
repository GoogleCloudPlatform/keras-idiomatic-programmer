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

# ResNeXt (50, 101, 152)
# Paper: https://arxiv.org/pdf/1611.05431.pdf

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ReLU, BatchNormalization, Add
from tensorflow.keras.layers import Concatenate, Dense, GlobalAveragePooling2D, Lambda
from tensorflow.keras.layers import Activation
from tensorflow.keras.regularizers import l2

import sys
sys.path.append('../')
from models_c import Composable

class ResNeXt(Composable):
    """ Construct a Residual Next Convolution Neural Network """
    # Meta-parameter: number of filters in, out and number of blocks
    groups = { 50 : [ { 'filters_in': 128,  'filters_out' : 256,  'n_blocks': 3 }, 
                      { 'filters_in': 256,  'filters_out' : 512,  'n_blocks': 4 }, 
                      { 'filters_in': 512,  'filters_out' : 1024, 'n_blocks': 6 }, 
                      { 'filters_in': 1024, 'filters_out' : 2048, 'n_blocks': 3 } ],	 # ResNeXt50
               101 :[ { 'filters_in': 128,  'filters_out' : 256,  'n_blocks': 3 }, 
                      { 'filters_in': 256,  'filters_out' : 512,  'n_blocks': 4 }, 
                      { 'filters_in': 512,  'filters_out' : 1024, 'n_blocks': 23 }, 
                      { 'filters_in': 1024, 'filters_out' : 2048, 'n_blocks': 3 } ],	 # ResNeXt101
               152 :[ { 'filters_in': 128,  'filters_out' : 256,  'n_blocks': 3 }, 
                      { 'filters_in': 256,  'filters_out' : 512,  'n_blocks': 8 }, 
                      { 'filters_in': 512,  'filters_out' : 1024, 'n_blocks': 36 }, 
                      { 'filters_in': 1024, 'filters_out' : 2048, 'n_blocks': 3 } ] 	 # ResNeXt152
             }
    
    # Meta-parameter: width of group convolution
    cardinality = 32

    _model = None

    def __init__(self, n_layers, cardinality=32, input_shape=(224, 224, 3), n_classes=1000, reg=l2(0.001),
                 init_weights='he_normal', relu=None):
        """ Construct a Residual Next Convolution Neural Network
            n_layers    : number of layers.
            cardinality : width of group convolution
            input_shape : the input shape
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
                raise Exception("ResNeXt: Invalid value for n_layers")
            groups = self.groups[n_layers]
        # user defined
        else:
            groups = n_layers
        
        # The input tensor
        inputs = Input(shape=input_shape)

        # The Stem Group
        x = self.stem(inputs, reg=reg)

        # The Learner
        x = self.learner(x, cardinality=cardinality, groups=groups, reg=reg)

        # The Classifier 
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
        """ Construct the Stem Convolution Group
            inputs : input vector
            reg    : kernel regularizer
        """
        reg = metaparameters['reg']

        x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', use_bias=False,
                   kernel_initializer=self.init_weights, kernel_regularizer=reg)(inputs)
        x = BatchNormalization()(x)
        x = Composable.ReLU(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
        return x
    
    def learner(self, x, **metaparameters):
        """ Construct the Learner
            x          : input to the learner
            groups     : list of groups: filters in, filters out, number of blocks
        """
        groups = metaparameters['groups']

        # First ResNeXt Group (not-strided)
        x = ResNeXt.group(x, strides=(1, 1), **groups.pop(0), **metaparameters)

        # Remaining ResNeXt groups
        for group in groups:
            x = ResNeXt.group(x, **group, **metaparameters)
        return x

    @staticmethod
    def group(x, strides=(2, 2), init_weights=None, **metaparameters):
        """ Construct a Residual group
            x          : input to the group
            strides    : whether its a strided convolution
            filters_in : number of filters  (channels) at the input convolution
            filters_out: number of filters (channels) at the output convolution
            n_blocks   : number of blocks in the group
            cardinality: width of group convolution
        """
        n_blocks = metaparameters['n_blocks']

        # Double the size of filters to fit the first Residual Group
        # Reduce feature maps by 75% (strides=2, 2) to fit the next Residual Group
        x = ResNeXt.projection_block(x, strides=strides, nit_weights=init_weights, **metaparameters)

        # Remaining blocks
        for _ in range(n_blocks):
            x = ResNeXt.identity_block(x, init_weights=init_weights, **metaparameters)	
        return x

    @staticmethod
    def identity_block(x, init_weights=None, **metaparameters):
        """ Construct a ResNeXT block with identity link
            x          : input to block
            filters_in : number of filters  (channels) at the input convolution
            filters_out: number of filters (channels) at the output convolution
            cardinality: width of group convolution
            reg        : kernel regularizer
        """
        filters_in  = metaparameters['filters_in']
        filters_out = metaparameters['filters_out']
        if 'cardinality' in metaparameters:
            cardinality = metaparameters['cardinality']
        else:
            cardinality = ResNeXt.cardinality
        if 'reg' in metaparameters:
            reg = metaparameters['reg']
        else:
            reg = ResNeXt.reg

        if init_weights is None:
            init_weights = ResNeXt.init_weights
            
        # Remember the input
        shortcut = x

        # Dimensionality Reduction
        x = Conv2D(filters_in, (1, 1), strides=(1, 1), padding='same', use_bias=False,
                   kernel_initializer=init_weights, kernel_regularizer=reg)(shortcut)
        x = BatchNormalization()(x)
        x = Composable.ReLU(x)

        # Cardinality (Wide) Layer (split-transform)
        filters_card = filters_in // cardinality
        groups = []
        for i in range(cardinality):
            group = Lambda(lambda z: z[:, :, :, i * filters_card:i * filters_card + filters_card])(x)
            groups.append(Conv2D(filters_card, (3, 3), strides=(1, 1), padding='same', use_bias=False,
                                 kernel_initializer=init_weights, kernel_regularizer=reg)(group))

        # Concatenate the outputs of the cardinality layer together (merge)
        x = Concatenate()(groups)
        x = BatchNormalization()(x)
        x = Composable.ReLU(x)

        # Dimensionality restoration
        x = Conv2D(filters_out, (1, 1), strides=(1, 1), padding='same', use_bias=False,
                   kernel_initializer=init_weights, kernel_regularizer=reg)(x)
        x = BatchNormalization()(x)

        # Identity Link: Add the shortcut (input) to the output of the block
        x = Add()([shortcut, x])
        x = Composable.ReLU(x)
        return x

    @staticmethod
    def projection_block(x, strides=(2, 2), init_weights=None, **metaparameters):
        """ Construct a ResNeXT block with projection shortcut
            x          : input to the block
            strides    : whether entry convolution is strided (i.e., (2, 2) vs (1, 1))
            filters_in : number of filters  (channels) at the input convolution
            filters_out: number of filters (channels) at the output convolution
            cardinality: width of group convolution
            reg        : kernel regularizer
        """
        filters_in = metaparameters['filters_in']
        filters_out = metaparameters['filters_out']
        if 'cardinality' in metaparameters:
            cardinality = metaparameters['cardinality']
        else:
            cardinality = ResNeXt.cardinality
        if 'reg' in metaparameters:
            reg = metaparameters['reg']
        else:
            reg = ResNeXt.reg

        if init_weights is None:
            init_weights = ResNeXt.init_weights
    
        # Construct the projection shortcut
        # Increase filters by 2X to match shape when added to output of block
        shortcut = Conv2D(filters_out, (1, 1), strides=strides, padding='same',
                          kernel_initializer=init_weights, kernel_regularizer=reg)(x)
        shortcut = BatchNormalization()(shortcut)

        # Dimensionality Reduction
        x = Conv2D(filters_in, (1, 1), strides=(1, 1), padding='same', use_bias=False,
                   kernel_initializer=init_weights, kernel_regularizer=reg)(x)
        x = BatchNormalization()(x)
        x = Composable.ReLU(x)

        # Cardinality (Wide) Layer (split-transform)
        filters_card = filters_in // cardinality
        groups = []
        for i in range(cardinality):
            group = Lambda(lambda z: z[:, :, :, i * filters_card:i * filters_card + filters_card])(x)
            groups.append(Conv2D(filters_card, (3, 3), strides=strides, padding='same', use_bias=False,
                                 kernel_initializer=init_weights, kernel_regularizer=reg)(group))

        # Concatenate the outputs of the cardinality layer together (merge)
        x = Concatenate()(groups)
        x = BatchNormalization()(x)
        x = Composable.ReLU(x)

        # Dimensionality restoration
        x = Conv2D(filters_out, (1, 1), strides=(1, 1), padding='same', use_bias=False,
                   kernel_initializer=init_weights, kernel_regularizer=reg)(x)
        x = BatchNormalization()(x)

        # Identity Link: Add the shortcut (input) to the output of the block
        x = Add()([shortcut, x])
        x = Composable.ReLU(x)
        return x
    
    def classifier(self, x, n_classes, **metaparameters):
        """ Construct the Classifier
            x         : input to the classifier
            n_classes : number of output classes
            reg       : kernel regularizer
        """
        reg = metaparameters['reg']

        # Save the encoding layer
        self.encoding = x

        # Final Dense Outputting Layer 
        x = GlobalAveragePooling2D()(x)

        # Save the bottleneck layer
        self.bottleneck = x
        
        x = Dense(n_classes, 
                        kernel_initializer=self.init_weights, kernel_regularizer=reg)(x)
        # Save the pre-activation probabilities layer
        self.probabilities = x
        outputs = Activation('softmax')(x)
        # Save the post-activation probabilities layer
        self.softmax = outputs
        return outputs


# Example
# resnext = ResNeXt(50)

