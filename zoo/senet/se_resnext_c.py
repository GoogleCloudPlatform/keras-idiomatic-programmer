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

# SE-ResNeXt (50, 101, 152)
# Trainable params: 27,547,688
# Paper: https://arxiv.org/pdf/1709.01507.pdf

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, ReLU, Dense, Add, Activation
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Reshape, Multiply, Lambda, Concatenate
from tensorflow.keras.regularizers import l2

import sys
sys.path.append('../')
from models_c import Composable

class SEResNeXt(Composable):
    """ Construct a Squeeze & Excite Residual Next Convolution Neural Network """
    # Meta-parameter: number of filters in, out and number of blocks
    groups = { 50 : [ { 'filters_in': 128,  'filters_out' : 256,  'n_blocks': 3 },
                      { 'filters_in': 256,  'filters_out' : 512,  'n_blocks': 4 },
                      { 'filters_in': 512,  'filters_out' : 1024, 'n_blocks': 6 },
                      { 'filters_in': 1024, 'filters_out' : 2048, 'n_blocks': 3 } ],     # SE-ResNeXt50
               101 :[ { 'filters_in': 128,  'filters_out' : 256,  'n_blocks': 3 },
                      { 'filters_in': 256,  'filters_out' : 512,  'n_blocks': 4 },
                      { 'filters_in': 512,  'filters_out' : 1024, 'n_blocks': 23 },
                      { 'filters_in': 1024, 'filters_out' : 2048, 'n_blocks': 3 } ],     # SE-ResNeXt101
               152 :[ { 'filters_in': 128,  'filters_out' : 256,  'n_blocks': 3 },
                      { 'filters_in': 256,  'filters_out' : 512,  'n_blocks': 8 },
                      { 'filters_in': 512,  'filters_out' : 1024, 'n_blocks': 36 },
                      { 'filters_in': 1024, 'filters_out' : 2048, 'n_blocks': 3 } ]      # SE-ResNeXt152
             }

    # Initial Hyperparameters
    hyperparameters = { 'initializer': 'he_normal',
                        'regularizer': l2(0.001),
                        'relu_clip'  : None,
                        'bn_epsilon' : None,
                        'use_bias'   : False
                      }

    def __init__(self, n_layers, cardinality=32, ratio=16, 
                 input_shape=(224, 224, 3), n_classes=1000, include_top=True,
                 **hyperparameters):
        """ Construct a Residual Next Convolution Neural Network
            n_layers    : number of layers
            cardinality : width of group convolution
            ratio       : amount of filter reduction in squeeze operation
            input_shape : the input shape
            n_classes   : number of output classes
            include_top : whether to include classifier
            initializer : kernel initializer
            regularizer : kernel regularization
            relu_clip   : max value for ReLU
            bn_epsilon  : epsilon for batch norm
            use_bias    : whether to use bias with batchnorm
        """
        # Configure base (super) class
        Composable.__init__(self, input_shape, include_top, self.hyperparameters, **hyperparameters)
        
        # predefined
        if isinstance(n_layers, int):
            if n_layers not in [50, 101, 152]:
                raise Exception("SE-ResNeXt: Invalid value for n_layers")
            groups = list(self.groups[n_layers])
        # user defined
        else:
            groups = n_layers

        # The input tensor
        inputs = Input(shape=input_shape)

        # The Stem Group
        x = self.stem(inputs)

        # The Learner
        outputs = self.learner(x, groups=groups, cardinality=cardinality, ratio=ratio)

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
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        return x

    def learner(self, x, **metaparameters):
        """ Construct the Learner
            x          : input to the learner
            groups     : list of groups: filters in, filters out, number of blocks
        """
        groups = metaparameters['groups']

        # First ResNeXt Group (not strided)
        x = self.group(x, strides=(1, 1), **groups.pop(0), **metaparameters)

        # Remaining ResNeXt Groups
        for group in groups:
            x = self.group(x, **group, **metaparameters)
        return x

    def group(self, x, strides=(2, 2), **metaparameters):
        """ Construct a Squeeze-Excite Group
            x          : input to the group
            strides    : whether projection block is strided
            n_blocks   : number of blocks in the group
        """
        n_blocks = metaparameters['n_blocks']

        # First block is a linear projection block
        x = self.projection_block(x, strides=strides, **metaparameters)

        # Remaining blocks are identity links
        for _ in range(n_blocks-1):
            x = self.identity_block(x, **metaparameters) 
        return x

    def squeeze_excite_block(self, x, **metaparameters):
        """ Construct a Squeeze and Excite block
            x     : input to the block
            ratio : amount of filter reduction during squeeze
        """  
        ratio = metaparameters['ratio']
            
        # Remember the input
        shortcut = x
    
        # Get the number of filters on the input
        filters = x.shape[-1]

        # Squeeze (dimensionality reduction)
        # Do global average pooling across the filters, which will output a 1D vector
        x = GlobalAveragePooling2D()(x)
    
        # Reshape into 1x1 feature maps (1x1xC)
        x = Reshape((1, 1, filters))(x)
    
        # Reduce the number of filters (1x1xC/r)
        x = self.Dense(x, filters // ratio, activation='relu', use_bias=False, **metaparameters)

        # Excitation (dimensionality restoration)
        # Restore the number of filters (1x1xC)
        x = self.Dense(x, filters, activation='sigmoid', use_bias=False, **metaparameters)

        # Scale - multiply the squeeze/excitation output with the input (WxHxC)
        x = Multiply()([shortcut, x])
        return x

    def identity_block(self, x, **metaparameters):
        """ Construct a ResNeXT block with identity link
            x          : input to block
            filters_in : number of filters  (channels) at the input convolution
            filters_out: number of filters (channels) at the output convolution
            cardinality: width of cardinality layer
        """ 
        filters_in  = metaparameters['filters_in']
        filters_out = metaparameters['filters_out']
        cardinality = metaparameters['cardinality']
    
        # Remember the input
        shortcut = x

        # Dimensionality Reduction
        x = self.Conv2D(x, filters_in, kernel_size=(1, 1), strides=(1, 1), padding='same', 
                        **metaparameters)
        x = self.BatchNormalization(x)
        x = self.ReLU(x)

        # Cardinality (Wide) Layer (split-transform)
        filters_card = filters_in // cardinality
        groups = []
        for i in range(cardinality):
            group = Lambda(lambda z: z[:, :, :, i * filters_card:i * filters_card + filters_card])(x)
            groups.append(self.Conv2D(group, filters_card, kernel_size=(3, 3), strides=(1, 1), padding='same', 
                                      **metaparameters))

        # Concatenate the outputs of the cardinality layer together (merge)
        x = Concatenate()(groups)
        x = self.BatchNormalization(x)
        x = self.ReLU(x)

        # Dimensionality restoration
        x = self.Conv2D(x, filters_out, kernel_size=(1, 1), strides=(1, 1), padding='same', 
                        **metaparameters)
        x = self.BatchNormalization(x)
    
        # Pass the output through the squeeze and excitation block
        x = self.squeeze_excite_block(x, **metaparameters)

        # Identity Link: Add the shortcut (input) to the output of the block
        x = Add()([shortcut, x])
        x = self.ReLU(x)
        return x

    def projection_block(self, x, strides=1, **metaparameters):
        """ Construct a ResNeXT block with projection shortcut
            x          : input to the block
            strides    : whether entry convolution is strided (i.e., (2, 2) vs (1, 1))
            filters_in : number of filters  (channels) at the input convolution
            filters_out: number of filters (channels) at the output convolution
            cardinality: width of cardinality layer
        """ 
        filters_in  = metaparameters['filters_in']
        filters_out = metaparameters['filters_out']
        cardinality = metaparameters['cardinality']
    
        # Construct the projection shortcut
        # Increase filters by 2X to match shape when added to output of block
        shortcut = self.Conv2D(x, filters_out, kernel_size=(1, 1), strides=strides, padding='same', 
                               **metaparameters)
        shortcut = self.BatchNormalization(shortcut)

        # Dimensionality Reduction
        x = self.Conv2D(x, filters_in, kernel_size=(1, 1), strides=(1, 1), padding='same', 
                        **metaparameters)
        x = self.BatchNormalization(x)
        x = self.ReLU(x)

        # Cardinality (Wide) Layer (split-transform)
        filters_card = filters_in // cardinality
        groups = []
        for i in range(cardinality):
            group = Lambda(lambda z: z[:, :, :, i * filters_card:i * filters_card + filters_card])(x)
            groups.append(self.Conv2D(group, filters_card, kernel_size=(3, 3), strides=strides, padding='same', 
                                      **metaparameters))

        # Concatenate the outputs of the cardinality layer together (merge)
        x = Concatenate()(groups)
        x = self.BatchNormalization(x)
        x = self.ReLU(x)

        # Dimensionality restoration
        x = self.Conv2D(x, filters_out, kernel_size=(1, 1), strides=(1, 1), padding='same', 
                        **metaparameters)
        x = self.BatchNormalization(x)
    
        # Pass the output through the squeeze and excitation block
        x = self.squeeze_excite_block(x, **metaparameters)

        # Add the projection shortcut (input) to the output of the block
        x = Add()([shortcut, x])
        x = self.ReLU(x)
        return x

# Example
senet = SEResNeXt(50)

def example():
    ''' Example for constructing/training a SE-ResNeXt model on CIFAR-10
    '''
    # Example of constructing a mini-SE-ResNeXt
    groups = [ { 'filters_in': 128,  'filters_out' : 256,  'n_blocks': 1 },
               { 'filters_in': 256,  'filters_out' : 512,  'n_blocks': 2 },
               { 'filters_in': 512,  'filters_out' : 1024, 'n_blocks': 2 } ]
    senet = SEResNeXt(groups, input_shape=(32, 32, 3), n_classes=10)
    senet.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
    senet.model.summary()
    senet.cifar10()

# example()
