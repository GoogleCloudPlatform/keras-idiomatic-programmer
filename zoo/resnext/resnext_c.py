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
# Trainable params: 31,010,344
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

    # Initial Hyperparameters
    hyperparameters = { 'initializer': 'he_normal',
                        'regularizer': l2(0.001),
                        'relu_clip'  : None,
                        'bn_epsilon' : None,
                        'use_bias'   : False
                      }
    
    def __init__(self, n_layers, cardinality=32, 
                 input_shape=(224, 224, 3), n_classes=1000, include_top=True,
                 **hyperparameters):
        """ Construct a Residual Next Convolution Neural Network
            n_layers    : number of layers.
            cardinality : width of group convolution
            input_shape : the input shape
            n_classes   : number of output classes
            include_top : whether to include classifier
            regularizer : kernel regularizer
            initializer : kernel initializer
            relu_clip   : max value for ReLU
            use_bias    : whether to use bias with batchnorm
        """
        # Configure base (super) class
        Composable.__init__(self, input_shape, include_top, self.hyperparameters, **hyperparameters)
        
        # predefined
        if isinstance(n_layers, int):
            if n_layers not in [50, 101, 152]:
                raise Exception("ResNeXt: Invalid value for n_layers")
            groups = list(self.groups[n_layers])
        # user defined
        else:
            groups = n_layers
        
        # The input tensor
        inputs = Input(shape=input_shape)

        # The Stem Group
        x = self.stem(inputs)

        # The Learner
        outputs = self.learner(x, cardinality=cardinality, groups=groups)

        # The Classifier 
        if include_top:
            # Add hidden dropout
            outputs = self.classifier(outputs, n_classes, dropout=0.0)

        # Instantiate the Model
        self._model = Model(inputs, outputs)

    def stem(self, inputs):
        """ Construct the Stem Convolution Group
            inputs : input vector
        """
        x = self.Conv2D(inputs, 64, (7, 7), strides=(2, 2), padding='same')
        x = self.BatchNormalization(x)
        x = self.ReLU(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
        return x
    
    def learner(self, x, **metaparameters):
        """ Construct the Learner
            x          : input to the learner
            groups     : list of groups: filters in, filters out, number of blocks
            cardinality: width of group convolution
        """
        cardinality = metaparameters['cardinality']
        groups      = metaparameters['groups']

        # First ResNeXt Group (not-strided)
        x = self.group(x, strides=(1, 1), **groups.pop(0), cardinality=cardinality)

        # Remaining ResNeXt groups
        for group in groups:
            x = self.group(x, **group, cardinality=cardinality)
        return x

    def group(self, x, strides=(2, 2), **metaparameters):
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
        x = self.projection_block(x, strides=strides, **metaparameters)

        # Remaining blocks
        for _ in range(n_blocks):
            x = self.identity_block(x, **metaparameters)	
        return x

    def identity_block(self, x, **metaparameters):
        """ Construct a ResNeXT block with identity link
            x           : input to block
            filters_in  : number of filters  (channels) at the input convolution
            filters_out : number of filters (channels) at the output convolution
            cardinality : width of group convolution
        """
        filters_in  = metaparameters['filters_in']
        filters_out = metaparameters['filters_out']
        cardinality = metaparameters['cardinality']
            
        # Remember the input
        shortcut = x

        # Dimensionality Reduction
        x = self.Conv2D(x, filters_in, (1, 1), strides=(1, 1), padding='same', **metaparameters)
        x = self.BatchNormalization(x)
        x = self.ReLU(x)

        # Cardinality (Wide) Layer (split-transform)
        filters_card = filters_in // cardinality
        groups = []
        for i in range(cardinality):
            group = Lambda(lambda z: z[:, :, :, i * filters_card:i * filters_card + filters_card])(x)
            groups.append(self.Conv2D(group, filters_card, (3, 3), strides=(1, 1), padding='same', 
                                      **metaparameters))

        # Concatenate the outputs of the cardinality layer together (merge)
        x = Concatenate()(groups)
        x = self.BatchNormalization(x)
        x = self.ReLU(x)

        # Dimensionality restoration
        x = self.Conv2D(x, filters_out, (1, 1), strides=(1, 1), padding='same', **metaparameters)
        x = self.BatchNormalization(x)

        # Identity Link: Add the shortcut (input) to the output of the block
        x = Add()([shortcut, x])
        x = self.ReLU(x)
        return x

    def projection_block(self, x, strides=(2, 2), **metaparameters):
        """ Construct a ResNeXT block with projection shortcut
            x          : input to the block
            strides    : whether entry convolution is strided (i.e., (2, 2) vs (1, 1))
            filters_in : number of filters  (channels) at the input convolution
            filters_out: number of filters (channels) at the output convolution
            cardinality: width of group convolution
        """
        filters_in = metaparameters['filters_in']
        filters_out = metaparameters['filters_out']
        cardinality = metaparameters['cardinality']
    
        # Construct the projection shortcut
        # Increase filters by 2X to match shape when added to output of block
        shortcut = self.Conv2D(x, filters_out, (1, 1), strides=strides, padding='same', **metaparameters)
        shortcut = self.BatchNormalization(shortcut)

        # Dimensionality Reduction
        x = self.Conv2D(x, filters_in, (1, 1), strides=(1, 1), padding='same', **metaparameters)
        x = self.BatchNormalization(x)
        x = self.ReLU(x)

        # Cardinality (Wide) Layer (split-transform)
        filters_card = filters_in // cardinality
        groups = []
        for i in range(cardinality):
            group = Lambda(lambda z: z[:, :, :, i * filters_card:i * filters_card + filters_card])(x)
            groups.append(self.Conv2D(group, filters_card, (3, 3), strides=strides, padding='same', 
                                      **metaparameters))

        # Concatenate the outputs of the cardinality layer together (merge)
        x = Concatenate()(groups)
        x = self.BatchNormalization(x)
        x = self.ReLU(x)

        # Dimensionality restoration
        x = self.Conv2D(x, filters_out, (1, 1), strides=(1, 1), padding='same', **metaparameters)
        x = self.BatchNormalization(x)

        # Identity Link: Add the shortcut (input) to the output of the block
        x = Add()([shortcut, x])
        x = self.ReLU(x)
        return x


# Example
# resnext = ResNeXt(50)

def example():
    ''' Example for constructing/training a ResNeXt model on CIFAR-10
    '''
    # Example of constructing a mini-ResNeXt
    groups =  [ { 'filters_in': 128,  'filters_out' : 256,  'n_blocks': 1 },
                { 'filters_in': 256,  'filters_out' : 512,  'n_blocks': 2 },
                { 'filters_in': 512,  'filters_out' : 1024, 'n_blocks': 2 } ]
    resnext = ResNeXt(groups, input_shape=(32, 32, 3), n_classes=10)
    resnext.model.summary()
    resnext.cifar10()

# example()
