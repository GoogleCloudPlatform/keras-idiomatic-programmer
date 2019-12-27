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

# DenseNet-BC 121/169/201 (2016)
# Paper: https://arxiv.org/pdf/1608.06993.pdf

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import ZeroPadding2D, Conv2D, MaxPooling2D, BatchNormalization, ReLU
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, AveragePooling2D, Concatenate

def stem(inputs, n_filters):
    """ Construct the Stem Convolution Group
        inputs   : input tensor
        n_filters: number of filters for the dense blocks (k)
    """
    # Pads input from 224x224 to 230x230
    x = ZeroPadding2D(padding=((3, 3), (3, 3)))(inputs)
    
    # First large convolution for abstract features for input 224 x 224 and output 112 x 112
    # Stem convolution uses 2 * k (growth rate) number of filters
    x = Conv2D(2 * n_filters, (7, 7), strides=(2, 2), use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # Add padding so when downsampling we fit shape 56 x 56
    x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = MaxPooling2D((3, 3), strides=2)(x)
    return x
    
def learner(x, groups, n_filters, reduction):
    """ Construct the Learner
        x         : input to the learner
        groups    : set of number of blocks per group
        n_filters : number of filters (growth rate)
        reduction : the amount to reduce (compress) feature maps by
    """
    # pop off the list the last dense block
    last = groups.pop()

    # Create the dense groups and interceding transition blocks
    for n_blocks in groups:
        x = group(x, n_blocks, n_filters, reduction)

    # Add the last dense group w/o a following transition block
    x = group(x, last, n_filters)
    return x

def group(x, n_blocks, n_filters, reduction=None):
    """ Construct a Dense Group
        x         : input to the group
        n_blocks  : number of residual blocks in dense block
        n_filters : number of filters in convolution layer in residual block
        reduction : amount to reduce feature maps by
    """
    # Construct a group of densely connected residual blocks
    for _ in range(n_blocks):
        x = dense_block(x, n_filters)

    # Construct interceding transition block
    if reduction is not None:
        x = trans_block(x, reduction)
    return x

def dense_block(x, n_filters):
    """ Construct a Densely Connected Residual Block
        x        : input to the block
        n_filters: number of filters in convolution layer in residual block
    """
    # Remember input tensor into residual block
    shortcut = x 
    
    # BN-RE-Conv pre-activation form of convolutions

    # Dimensionality expansion, expand filters by 4 (DenseNet-B)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(4 * n_filters, (1, 1), strides=(1, 1), use_bias=False, kernel_initializer='he_normal')(x)
    
    # Bottleneck convolution
    # 3x3 convolution with padding=same to preserve same shape of feature maps
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(n_filters, (3, 3), strides=(1, 1), padding='same', use_bias=False, kernel_initializer='he_normal')(x)

    # Concatenate the input (identity) with the output of the residual block
    # Concatenation (vs. merging) provides Feature Reuse between layers
    x = Concatenate()([shortcut, x])
    return x

def trans_block(x, reduction):
    """ Construct a Transition Block
        x        : input layer
        reduction: percentage of reduction of feature maps
    """

    # Reduce (compress) the number of feature maps (DenseNet-C)
    # shape[n] returns a class object. We use int() to cast it into the dimension size
    n_filters = int( int(x.shape[3]) * reduction)
    
    # BN-LI-Conv pre-activation form of convolutions

    # Use 1x1 linear projection convolution
    x = BatchNormalization()(x)
    x = Conv2D(n_filters, (1, 1), strides=(1, 1), use_bias=False, kernel_initializer='he_normal')(x)

    # Use mean value (average) instead of max value sampling when pooling reduce by 75%
    x = AveragePooling2D((2, 2), strides=(2, 2))(x)
    return x
    
def classifier(x, n_classes):
    """ Construct the Classifier Group
        x         : input to the classifier
        n_classes : number of output classes
    """
    # Global Average Pooling will flatten the 7x7 feature maps into 1D feature maps
    x = GlobalAveragePooling2D()(x)
    # Fully connected output layer (classification)
    x = Dense(n_classes, activation='softmax', kernel_initializer='he_normal')(x)
    return x
    
# Meta-parameter: amount to reduce feature maps by (compression factor) during transition blocks
reduction = 0.5

# Meta-parameter: number of filters in a convolution block within a residual block (growth rate)
n_filters = 32

# Meta-parameter: number of residual blocks in each dense group
groups = { 121 : [6, 12, 24, 16],	# DenseNet 121
           169 : [6, 12, 32, 32],	# DenseNet 169
           201 : [6, 12, 48, 32] }	# DenseNet 201


# The input vector
inputs = Input(shape=(224, 224, 3))

# The Stem Convolution Group
x = stem(inputs, n_filters)

# The Learner
x = learner(x, groups[121], n_filters, reduction)

# Classifier for 1000 classes
outputs = classifier(x, 1000)

# Instantiate the model
model = Model(inputs, outputs)
