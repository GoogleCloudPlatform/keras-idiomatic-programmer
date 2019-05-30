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

# Dense Net 121
from keras import layers, Input, Model

def stem(inputs):
    """ The Stem Convolution Group
        inputs : input tensor
    """
    # First large convolution for abstract features for input 230 x 230 and output
    # 112 x 112
    x = layers.Conv2D(64, (7, 7), strides=2)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    # Add padding so when downsampling we fit shape 56 x 56
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = layers.MaxPooling2D((3, 3), strides=2)(x)
    return x

def dense_block(x, nblocks, nb_filters):
    """ Construct a Dense Block
        x         : input layer
        nblocks   : number of residual blocks in dense block
        nb_filters: number of filters in convolution layer in residual block
    """
    # Construct a group of residual blocks
    for _ in range(nblocks):
        x = residual_block(x, nb_filters)
    return x

def residual_block(x, nb_filters):
    """ Construct Residual Block
        x         : input layer
        nb_filters: number of filters in convolution layer in residual block
    """
    shortcut = x # remember input tensor into residual block

    # Bottleneck convolution, expand filters by 4 (DenseNet-B)
    x = layers.Conv2D(4 * nb_filters, (1, 1), strides=(1, 1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # 3 x 3 convolution with padding=same to preserve same shape of feature maps
    x = layers.Conv2D(nb_filters, (3, 3), strides=(1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Concatenate the input (identity) with the output of the residual block
    # Concatenation (vs. merging) provides Feature Reuse between layers
    x = layers.concatenate([shortcut, x])
    return x

def trans_block(x, reduce_by):
    """ Construct a Transition Block
        x        : input layer
        reduce_by: percentage of reduction of feature maps
    """

    # Reduce (compression) the number of feature maps (DenseNet-C)
    # shape[n] returns a class object. We use int() to cast it into the dimension
    # size
    nb_filters = int( int(x.shape[3]) * reduce_by )

    # Bottleneck convolution
    x = layers.Conv2D(nb_filters, (1, 1), strides=(1, 1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Use mean value (average) instead of max value sampling when pooling
    # reduce by 75%
    x = layers.AveragePooling2D((2, 2), strides=(2, 2))(x)
    return x

inputs = Input(shape=(230, 230, 3))

# Create the Stem Convolution Group
x = stem(inputs)

# number of residual blocks in each dense block
blocks = [6, 12, 24, 16]

# pop off the list the last dense block
last   = blocks.pop()

# amount to reduce feature maps by (compression) during transition blocks
reduce_by = 0.5

# number of filters in a convolution block within a residual block
nb_filters = 32

# Create the dense blocks and interceding transition blocks
for nblocks in blocks:
    x = dense_block(x, nblocks, nb_filters)
    x = trans_block(x, reduce_by)

# Add the last dense block w/o a following transition block
x = dense_block(x, last, nb_filters)

# Classifier
# Global Average Pooling will flatten the 7x7 feature maps into 1D feature maps
x = layers.GlobalAveragePooling2D()(x)
# Fully connected output layer (classification)
x = layers.Dense(1000, activation='softmax')(x)

model = Model(inputs, x)
model.summary()
