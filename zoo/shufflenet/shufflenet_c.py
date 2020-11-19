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

# ShuffleNet v1.0 (composable)
# Trainable params: 1,680,620
# Paper: https://arxiv.org/pdf/1707.01083.pdf

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, ReLU, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Add, Concatenate, AveragePooling2D, DepthwiseConv2D, Lambda, Activation
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2

import sys
sys.path.append('../')
from models_c import Composable

class ShuffleNet(Composable):
    ''' Construct a Shuffle Convolution Neural Network '''
    # meta-parameter: number of shuffle blocks per shuffle group
    groups  = [ { 'n_blocks' : 4 }, { 'n_blocks' : 8 }, { 'n_blocks' : 4 } ]

    # meta-parameter: number of groups to partition feature maps (key), and
    # corresponding number of output filters (value)
    filters = {
            1: [{ 'n_filters' : 144 }, { 'n_filters' : 288 }, { 'n_filters' : 576 }],
            2: [{ 'n_filters' : 200 }, { 'n_filters' : 400 }, { 'n_filters' : 800 }],
            3: [{ 'n_filters' : 240 }, { 'n_filters' : 480 }, { 'n_filters' : 960 }],
            4: [{ 'n_filters' : 272 }, { 'n_filters' : 544 }, { 'n_filters' : 1088 }],
            8: [{ 'n_filters' : 384 }, { 'n_filters' : 768 }, { 'n_filters' : 1536 }]
    }

    # Initial Hyperparameters
    hyperparameters = { 'initializer': 'glorot_uniform',
                        'regularizer': l2(0.001),
                        'relu_clip'  : None,
                        'bn_epsilon' : None,
                        'use_bias'   : False
                      }

    def __init__(self, groups=None, filters=None, n_partitions=2, reduction=0.25, 
                 input_shape=(224, 224, 3), n_classes=1000, include_top=True,
                 **hyperparameters):
        ''' Construct a Shuffle Convolution Neural Network
            groups      : number of shuffle blocks per shuffle group
            filters     : filters per group based on partitions
            n_partitions: number of groups to partition the filters (channels)
            reduction   : dimensionality reduction on entry to a shuffle block
            input_shape : the input shape to the model
            n_classes   : number of output classes
            include_top : whether to include classifier
            initializer : kernel initializer
            regularizer : kernel regularizer
            relu_clip   : max value for ReLU
            bn_epsilon  : epsilon for batch norm
            use_bias    : whether to use bias in conjunction with batch norm
        '''
        Composable.__init__(self, input_shape, include_top, self.hyperparameters, **hyperparameters)
        
        if groups is None:
            groups = list(ShuffleNet.groups)

        if filters is None:
            filters = self.filters[n_partitions]

        # input tensor
        inputs = Input(shape=input_shape)

        # The Stem convolution group (referred to as Stage 1)
        x = self.stem(inputs)

        # The Learner
        outputs = self.learner(x, groups=groups, n_partitions=n_partitions, filters=filters, reduction=reduction)

        # The Classifier
        if include_top:
            # Add hidden dropout to classifier
            outputs = self.classifier(outputs, n_classes, dropout=0.0)

        self._model = Model(inputs, outputs)

    def stem(self, inputs):
        ''' Construct the Stem Convolution Group 
            inputs : input image (tensor)
        '''
        x = self.Conv2D(inputs, 24, (3, 3), strides=2, padding='same')
        x = self.BatchNormalization(x)
        x = self.ReLU(x)
        x = MaxPooling2D((3, 3), strides=2, padding='same')(x)
        return x

    def learner(self, x, **metaparameters):
        ''' Construct the Learner
            x            : input to the learner
            groups       : number of shuffle blocks per shuffle group
            n_parttitions: number of groups to partition feature maps (channels) into.
            filters      : number of filters per shuffle group
            reduction    : dimensionality reduction on entry to a shuffle block
        '''
        groups  = metaparameters['groups']
        filters = metaparameters['filters']

        # Assemble the shuffle groups
        for group in groups:
            x = self.group(x, **group, **filters.pop(0), **metaparameters)
        return x

    def group(self, x, **metaparameters):
        ''' Construct a Shuffle Group 
            x           : input to the group
            n_partitions: number of groups to partition feature maps (channels) into.
            n_blocks    : number of shuffle blocks for this group
        '''
        n_blocks     = metaparameters['n_blocks']
            
        # first block is a strided shuffle block
        x = self.strided_shuffle_block(x, **metaparameters)
    
        # remaining shuffle blocks in group
        for _ in range(n_blocks-1):
            x = self.shuffle_block(x, **metaparameters)
        return x
    
    def strided_shuffle_block(self, x, **metaparameters):
        ''' Construct a Strided Shuffle Block 
            x           : input to the block
            n_partitions: number of groups to partition feature maps (channels) into.
            n_filters   : number of filters
            reduction   : dimensionality reduction factor (e.g, 0.25)
        '''
        n_partitions = metaparameters['n_partitions']
        n_filters    = metaparameters['n_filters']
        reduction    = metaparameters['reduction']
        del metaparameters['n_filters']
        del metaparameters['n_partitions']
            
        # projection shortcut
        shortcut = x
        shortcut = AveragePooling2D((3, 3), strides=2, padding='same')(shortcut)   
    
        # On entry block, we need to adjust the number of output filters
        # of the entry pointwise group convolution to match the exit
        # pointwise group convolution, by subtracting the number of input filters
        n_filters -= int(x.shape[3])
    
        # pointwise group convolution, with dimensionality reduction
        x = self.pw_group_conv(x, n_partitions, int(reduction * n_filters), **metaparameters)
        x = self.ReLU(x)
    
        # channel shuffle layer
        x = self.channel_shuffle(x, n_partitions)

        # Depthwise 3x3 Strided Convolution
        x = self.DepthwiseConv2D(x, (3, 3), strides=2, padding='same', **metaparameters)
        x = self.BatchNormalization(x)

        # pointwise group convolution, with dimensionality restoration
        x = self.pw_group_conv(x, n_partitions, n_filters, **metaparameters)
    
        # Concatenate the projection shortcut to the output
        x = Concatenate()([shortcut, x])
        x = self.ReLU(x)
        return x

    def shuffle_block(self, x, **metaparameters):
        ''' Construct a shuffle Shuffle block  
            x           : input to the block
            n_partitions: number of groups to partition feature maps (channels) into.
            n_filters   : number of filters
            reduction   : dimensionality reduction factor (e.g, 0.25)
            reg         : kernel regularizer
        '''
        n_partitions = metaparameters['n_partitions']
        n_filters    = metaparameters['n_filters']
        reduction    = metaparameters['reduction']
        del metaparameters['n_filters']
        del metaparameters['n_partitions']
            
        # identity shortcut
        shortcut = x
    
        # pointwise group convolution, with dimensionality reduction
        x = self.pw_group_conv(x, n_partitions, int(reduction * n_filters), **metaparameters)
        x = self.ReLU(x)
    
        # channel shuffle layer
        x = self.channel_shuffle(x, n_partitions)
    
        # Depthwise 3x3 Convolution
        x = self.DepthwiseConv2D(x, (3, 3), strides=1, padding='same', **metaparameters)
        x = BatchNormalization()(x)
    
        # pointwise group convolution, with dimensionality restoration
        x = self.pw_group_conv(x, n_partitions, n_filters, **metaparameters)
    
        # Add the identity shortcut (input added to output)
        x = Add()([shortcut, x])
        x = self.ReLU(x)
        return x

    def pw_group_conv(self, x, n_partitions, n_filters, **metaparameters):
        ''' A Pointwise Group Convolution  
            x           : input tensor
            n_partitions: number of groups to partition feature maps (channels) into.
            n_filters   : number of filters
        '''
            
        # Calculate the number of input filters (channels)
        in_filters = x.shape[3]

        # Derive the number of input filters (channels) per group
        grp_in_filters  = in_filters // n_partitions
        # Derive the number of output filters per group (Note the rounding up)
        grp_out_filters = int(n_filters / n_partitions + 0.5)
      
        # Perform convolution across each channel group
        groups = []
        for i in range(n_partitions):
            # Slice the input across channel group
            group = Lambda(lambda x: x[:, :, :, grp_in_filters * i: grp_in_filters * (i + 1)])(x)

            # Perform convolution on channel group
            conv = self.Conv2D(group, grp_out_filters, (1,1), padding='same', strides=1, 
                               **metaparameters)
            # Maintain the point-wise group convolutions in a list
            groups.append(conv)

        # Concatenate the outputs of the group pointwise convolutions together
        x = Concatenate()(groups)
        # Do batch normalization of the concatenated filters (feature maps)
        x = self.BatchNormalization(x)
        return x

    def channel_shuffle(self, x, n_partitions):
        ''' Implements the channel shuffle layer 
            x            : input tensor
            n_partitions : number of groups to partition feature maps (channels) into.
        '''

        # Get dimensions of the input tensor
        batch, height, width, n_filters = x.shape

        # Derive the number of input filters (channels) per group
        grp_in_filters  = n_filters // n_partitions

        # Separate out the channel groups
        x = Lambda(lambda z: K.reshape(z, [-1, height, width, n_partitions, grp_in_filters]))(x)
        # Transpose the order of the channel groups (i.e., 3, 4 => 4, 3)
        x = Lambda(lambda z: K.permute_dimensions(z, (0, 1, 2, 4, 3)))(x)
        # Restore shape
        x = Lambda(lambda z: K.reshape(z, [-1, height, width, n_filters]))(x)
        return x
    
# Example
# shufflenet = ShuffleNet()

def example():
    ''' Example for constructing/training a ShuffleNet model on CIFAR-10
    '''
    # Example of constructing a mini-ShuffleNet
    groups  = [ { 'n_blocks' : 2 }, { 'n_blocks' : 4 } ]
    shufflenet = ShuffleNet(groups, input_shape=(32, 32, 3), n_classes=10)
    shufflenet.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
    shufflenet.model.summary()
    shufflenet.cifar10()

# example()
