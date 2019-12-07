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

# DenseNet-BC 121/169/201 + composable (2016)
# Paper: https://arxiv.org/pdf/1608.06993.pdf

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import ZeroPadding2D, Conv2D, MaxPooling2D, BatchNormalization, ReLU
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, AveragePooling2D, Concatenate
from tensorflow.keras.layers import Activation
from tensorflow.keras.regularizers import l2

import sys
sys.path.append('../')
from models_c import Composable

class DenseNet(Composable):
    """ Construct a Densely Connected Convolution Neural Network """
    # Meta-parameter: number of residual blocks in each dense group
    groups = { 121 : [ { 'n_blocks': 6 }, { 'n_blocks': 12 }, { 'n_blocks': 24 }, { 'n_blocks': 16 } ], # DenseNet 121
               169 : [ { 'n_blocks': 6 }, { 'n_blocks': 12 }, { 'n_blocks': 32 }, { 'n_blocks': 32 } ], # DenseNet 169
               201 : [ { 'n_blocks': 6 }, { 'n_blocks': 12 }, { 'n_blocks': 48 }, { 'n_blocks': 32 } ]  # DenseNet 201
	     }

    # Meta-parameter: amount to reduce feature maps by (compression factor) during transition blocks
    reduction = 0.5
    # Meta-parameter: number of filters in a convolution block within a residual block (growth rate)
    n_filters = 32

    def __init__(self, n_layers, n_filters=32, reduction=0.5, input_shape=(224, 224, 3), n_classes=1000,
                 reg=l2(0.001), init_weights='he_normal', relu=None):
        """ Construct a Densely Connected Convolution Neural Network
            n_layers    : number of layers
            n_filters   : number of filters (growth rate)
            reduction   : anount to reduce feature maps by (compression factor)
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
            if n_layers not in [121, 169, 201]:
                raise Exception("DenseNet: Invalid value for n_layers")
            groups = list(self.groups[n_layers])
        # user defined
        else:
            groups = n_layers

        # The input vector
        inputs = Input(shape=input_shape)

        # The Stem Convolution Group
        x = self.stem(inputs, n_filters)

        # The Learner
        x = self.learner(x, n_filters=n_filters, reduction=reduction, groups=groups)

        # The Classifier 
        outputs = self.classifier(x, n_classes)

        # Instantiate the model
        self._model = Model(inputs, outputs)

    def stem(self, inputs, n_filters):
        """ Construct the Stem Convolution Group
            inputs   : input tensor
            n_filters: number of filters for the dense blocks (k)
        """
        # Pads input from 224x224 to 230x230
        x = ZeroPadding2D(padding=((3, 3), (3, 3)))(inputs)
    
        # First large convolution for abstract features for input 224 x 224 and output 112 x 112
        # Stem convolution uses 2 * k (growth rate) number of filters
        x = Conv2D(2 * n_filters, (7, 7), strides=(2, 2), use_bias=False, 
                   kernel_initializer=self.init_weights, kernel_regularizer=self.reg)(x)
        x = BatchNormalization()(x)
        x = Composable.ReLU(x)
    
        # Add padding so when downsampling we fit shape 56 x 56
        x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
        x = MaxPooling2D((3, 3), strides=2)(x)
        return x
    
    def learner(self, x, **metaparameters):
        """ Construct the Learner
            x         : input to the learner
            groups    : set of number of blocks per group
        """
        groups = metaparameters['groups']

        # pop off the list the last dense block
        last = groups.pop()

        # Create the dense groups and interceding transition blocks
        for group in groups:
            x = DenseNet.group(x, **group, **metaparameters)

        # Add the last dense group w/o a following transition block
        metaparameters['reduction'] = None
        x = DenseNet.group(x, **last, **metaparameters)
        return x

    @staticmethod
    def group(x, **metaparameters):
        """ Construct a Dense Block
            x         : input to the block
            n_blocks  : number of residual blocks in dense block
            reduction : amount to reduce (compress) feature maps by
        """
        n_blocks  = metaparameters['n_blocks']
        reduction = metaparameters['reduction']

        # Construct a group of residual blocks
        for _ in range(n_blocks):
            x = DenseNet.residual_block(x, **metaparameters)

        # Construct interceding transition block
        if reduction is not None:
            x = DenseNet.trans_block(x, reduction=reduction)
        return x

    @staticmethod
    def residual_block(x, **metaparameters):
        """ Construct a Residual Block
            x        : input to the block
            n_filters: number of filters in convolution layer in residual block
            reg      : kernel regularizer
        """
        if 'n_filters' in metaparameters:
            n_filters = metaparameters['n_filters']
        else:
            n_filters = DenseNet.n_filters
        if 'reg' in metaparameters:
            reg = metaparameters['reg']
        else:
            reg = DenseNet.reg
        if 'init_weights' in metaparameters:
            init_weights = metaparameters['init_weights']
        else:
            init_weights = DenseNet.init_weights
            
        # Remember input tensor into residual block
        shortcut = x 
    
        # BN-RE-Conv pre-activation form of convolutions

        # Dimensionality expansion, expand filters by 4 (DenseNet-B)
        x = BatchNormalization()(x)
        x = Composable.ReLU(x)
        x = Conv2D(4 * n_filters, (1, 1), strides=(1, 1), use_bias=False, 
                   kernel_initializer=init_weights, kernel_regularizer=reg)(x)
    
        # Bottleneck convolution
        # 3x3 convolution with padding=same to preserve same shape of feature maps
        x = BatchNormalization()(x)
        x = Composable.ReLU(x)
        x = Conv2D(n_filters, (3, 3), strides=(1, 1), padding='same', use_bias=False, 
                   kernel_initializer=init_weights, kernel_regularizer=reg)(x)

        # Concatenate the input (identity) with the output of the residual block
        # Concatenation (vs. merging) provides Feature Reuse between layers
        x = Concatenate()([shortcut, x])
        return x

    @staticmethod
    def trans_block(x, **metaparameters):
        """ Construct a Transition Block
            x        : input layer
            reduction: percentage of reduction of feature maps
            reg      : kernel regularizer
        """
        if 'reduction' in metaparameters:
            reduction = metaparameters['reduction']
        else:
            reduction = DenseNet.reduction
        if 'reg' in metaparameters:
            reg = metaparameters['reg']
        else:
            reg = DenseNet.reg
        if 'init_weights' in metaparameters:
            init_weights = metaparameters['init_weights']
        else:
            init_weights = DenseNet.init_weights

        # Reduce (compress) the number of feature maps (DenseNet-C)
        # shape[n] returns a class object. We use int() to cast it into the dimension size
        n_filters = int( int(x.shape[3]) * reduction)
    
        # BN-LI-Conv pre-activation form of convolutions

        # Use 1x1 linear projection convolution
        x = BatchNormalization()(x)
        x = Conv2D(n_filters, (1, 1), strides=(1, 1), use_bias=False, 
                   kernel_initializer=init_weights, kernel_regularizer=reg)(x)

        # Use mean value (average) instead of max value sampling when pooling reduce by 75%
        x = AveragePooling2D((2, 2), strides=(2, 2))(x)
        return x
    
    def classifier(self, x, n_classes):
        """ Construct the Classifier Group
            x         : input to the classifier
            n_classes : number of output classes
        """
        # Save the encoding layer
        self.encoding = x

        # Global Average Pooling will flatten the 7x7 feature maps into 1D feature maps
        x = GlobalAveragePooling2D()(x)

        # Save the embeddinglayer
        self.embedding = x
        
        # Fully connected output layer (classification)
        x = Dense(n_classes,
                  kernel_initializer=self.init_weights, kernel_regularizer=self.reg)(x)
        # Save the pre-activation probabilities layer
        self.probabilities = x
        outputs = Activation('softmax')(x)
        # Save the post-activation probabilities layer
        self.softmax = outputs
        return outputs
    
# Example
# densenet = DenseNet(121)
