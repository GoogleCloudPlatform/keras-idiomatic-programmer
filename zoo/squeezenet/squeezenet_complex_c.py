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

# SqueezeNet v1.0 with complex bypass (i.e., transition convolution on identify link) (2016)
# Paper: https://arxiv.org/pdf/1602.07360.pdf

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Concatenate, Add, Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D, Activation
from tensorflow.keras.regularizers import l2

import sys
sys.path.append('../')
from models_c import Composable

class SqueezeNetComplex(Composable):
    ''' Construct a SqueezeNet Complex Bypass Convolution Neural Network '''
    # Meta-parameter: number of blocks and filters per group
    groups = [ [ { 'n_filters' : 16 }, { 'n_filters' : 16 }, { 'n_filters' : 32 } ], 
               [ { 'n_filters' : 32 }, { 'n_filters' : 48 }, { 'n_filters' : 48 },  { 'n_filters': 64 } ], 
               [ { 'n_filters' : 64 } ] ]

    # Initial Hyperparameters
    hyperparameters = { 'initializer': 'glorot_uniform',
                        'regularizer': l2(0.001),
                        'relu_clip'  : None,
                        'bn_epsilon' : None,
                        'use_bias'   : True
                      }

    def __init__(self, groups=None, dropout=0.5, 
                 input_shape=(224, 224, 3), n_classes=1000, include_top=True,
                 **hyperparameters):
        ''' Construct a SqueezeNet Complex Bypass Convolution Neural Network
            groups      : number of blocks/filters per group
            dropout     : percent of dropoput
            input_shape : input shape to model
            n_classes   : number of output classes
            include_top : whether to include classifier
            init_weights: kernel initializer
            reg         : kernel regularizer
            relu        : max value for ReLU
            bias        : whether to use bias in conjunction with batch norm
        '''
        Composable.__init__(self, input_shape, include_top, self.hyperparameters, **hyperparameters)
        
        if groups is None:
            groups = list(SqueezeNetComplex.groups)
            
        # The input shape
        inputs = Input(shape=input_shape)

        # The Stem Group
        x = self.stem(inputs)

        # The Learner
        outputs = self.learner(x, groups=groups, dropout=dropout)

        # The Classifier
        if include_top:
            outputs = self.classifier(outputs, n_classes)

        self._model = Model(inputs, outputs)
        
    def stem(self, inputs):
        ''' Construct the Stem Group
            inputs : input tensor
        '''      
        x = self.Conv2D(inputs, 96, (7, 7), strides=2, padding='same')
        x = self.ReLU(x)
        x = MaxPooling2D(3, strides=2)(x)
        return x

    def learner(self, x, **metaparameters):
        ''' Construct the Learner
            x      : input to the learner
            gropups: number of blocks/filters per group
            dropout: percent of dropout
        '''
        groups  = metaparameters['groups']
        dropout = metaparameters['dropout']

        last = groups.pop()
        
        # Add fire groups, progressively increase number of filters
        for group in groups:
        	x = self.group(x, blocks=group, **metaparameters)

        # Last fire block (module)
        x = self.fire_block(x, **last[0], **metaparameters)

        # Dropout is delayed to end of fire modules
        x = Dropout(dropout)(x)
        return x

    def group(self, x, **metaparameters):
        ''' Construct a Fire Group
            x      : input to the group
            blocks: list of number of filters per fire block (module)
        '''
        blocks = metaparameters['blocks']
            
        # Add the fire blocks (modules) for this group
        for block in blocks:
            x = self.fire_block(x, **block, **metaparameters)

        # Delayed downsampling
        x = MaxPooling2D((3, 3), strides=2)(x)
        return x

    def fire_block(self, x, **metaparameters):
        ''' Construct a Fire Block  with complex bypass
            x        : input to the block
            n_filters: number of filters in block
        '''
        n_filters = metaparameters['n_filters']
            
        # remember the input (identity)
        shortcut = x

        # if the number of input filters does not equal the number of output filters, then use
        # a transition convolution to match the number of filters in identify link to output
        if shortcut.shape[3] != 8 * n_filters:
            shortcut = self.Conv2D(shortcut, n_filters * 8, (1, 1), strides=1,  padding='same')
            shortcut = self.ReLU(shortcut)

        # squeeze layer
        squeeze = self.Conv2D(x, n_filters, (1, 1), strides=1, padding='same')
        squeeze = self.ReLU(squeeze)

        # branch the squeeze layer into a 1x1 and 3x3 convolution and double the number
        # of filters
        expand1x1 = self.Conv2D(squeeze, n_filters * 4, (1, 1), strides=1, padding='same')
        expand1x1 = self.ReLU(expand1x1)
        expand3x3 = self.Conv2D(squeeze, n_filters * 4, (3, 3), strides=1, padding='same')
        expand3x3 = self.ReLU(expand3x3)

        # concatenate the feature maps from the 1x1 and 3x3 branches
        x = Concatenate()([expand1x1, expand3x3])
    
        # if identity link, add (matrix addition) input filters to output filters
        if shortcut is not None:
            x = Add()([x, shortcut])
        return x

    def classifier(self, x, n_classes):
        ''' Construct the Classifier 
            x        : input to the classifier
            n_classes: number of output classes
        '''
        # Save the encoding layer
        self.encoding = x
        
        # set the number of filters equal to number of classes
        x = self.Conv2D(x, n_classes, (1, 1), strides=1,  padding='same')
        x = self.ReLU(x)
        # reduce each filter (class) to a single value
        x = GlobalAveragePooling2D()(x)

        # Save the pre-activation probabilities
        self.probabilities = x
        outputs = Activation('softmax')(x)
        # Save the post-activations probabilities
        self.softmax = outputs
        return outputs

# Example
# squeezenet = SqueezeNetComplex()

def example():
    ''' Example for constructing/training a SqueezeNet Complex  model on CIFAR-10
    '''
    # Example of constructing a mini-SqueezeNet
    groups = [ [ { 'n_filters' : 16 }, { 'n_filters' : 16 }, { 'n_filters' : 32 } ],
               [ { 'n_filters' : 64 } ] ]
    squeezenet = SqueezeNetComplex(groups, input_shape=(32, 32, 3), n_classes=10)
    squeezenet.model.summary()
    squeezenet.cifar10()

# example()
