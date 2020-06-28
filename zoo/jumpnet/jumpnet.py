# Copyright 2020 Google LLC
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

# JumpNet (50, 101, 152)
# Residual Groups and Blocks are ResNet v2 - w/o projection block
# Stem convolution is a stack of two 3x3 filters (factorized 5x5), as in Inception v3

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, ReLU
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Add, Concatenate
from tensorflow.keras.regularizers import l2

def stem(inputs):
    """ Construct the Stem Convolutional Group 
        inputs : the input vector
    """
    
    # Stack of two 3x3 filters
    x = Conv2D(32, (3, 3), strides=(2, 2), padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(reg))(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(reg))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    return x

def learner(x, groups):
    """ Construct the Learner
        x     : input to the learner
        groups: list of groups: number of filters and blocks
    """
    # Residual Groups
    for n_filters, n_blocks in groups:
        x = group(x, n_filters, n_blocks)
    return x
    
def group(x, n_filters, n_blocks):
    """ Construct a Residual Group
        x         : input into the group
        n_filters : number of filters for the group
        n_blocks  : number of residual blocks with identity link
    """
    # Save the input to the group for the jump link at the end.
    shortcut = BatchNormalization()(x)
    shortcut = Conv2D(n_filters, (1, 1), strides=(2, 2),
               kernel_initializer='he_normal', kernel_regularizer=l2(reg))(shortcut)
    
    # Identity residual blocks
    for _ in range(n_blocks):
        x = identity_block(x, n_filters)

    # Feature Pooling at the end of the group
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(n_filters, (1, 1), strides=(2, 2),
               kernel_initializer='he_normal', kernel_regularizer=l2(reg))(x)

    # Construct the jumpn link
    x = Concatenate()([shortcut, x])
    return x

def identity_block(x, n_filters):
    """ Construct a Bottleneck Residual Block with Identity Link
        x        : input into the block
        n_filters: number of filters
    """
    
    # Save input vector (feature maps) for the identity link
    shortcut = x
    
    ## Construct the 1x1, 3x3, 1x1 convolution block
    
    # Dimensionality reduction
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(n_filters, (1, 1), strides=(1, 1),
               kernel_initializer='he_normal', kernel_regularizer=l2(reg))(x)

    # Bottleneck layer
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(n_filters, (3, 3), strides=(1, 1), padding="same",
               kernel_initializer='he_normal', kernel_regularizer=l2(reg))(x)

    # no dimensionality restoration
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(n_filters, (1, 1), strides=(1, 1),
               kernel_initializer='he_normal', kernel_regularizer=l2(reg))(x)

    # Add the identity link (input) to the output of the residual block
    x = Add()([shortcut, x])
    return x


def classifier(x, n_classes):
    """ Construct the Classifier Group 
        x         : input to the classifier
        n_classes : number of output classes
    """
    # Pool at the end of all the convolutional residual blocks
    x = GlobalAveragePooling2D()(x)

    # Final Dense Outputting Layer for the outputs
    outputs = Dense(n_classes, activation='softmax', kernel_initializer='he_normal')(x)
    return outputs

# Meta-parameter: list of groups: number of filters and number of blocks
groups = { 50 : [ (64, 3), (128, 4), (256, 6),  (512, 3) ],           # ResNet50
           101: [ (64, 3), (128, 4), (256, 23), (512, 3) ],           # ResNet101
           152: [ (64, 3), (128, 8), (256, 36), (512, 3) ]            # ResNet152
         }

# L2 regularization (weight decay)
reg = 0.001

# The input tensor
inputs = Input(shape=(224, 224, 3))

# The stem convolutional group
x = stem(inputs)

# The learner
x = learner(x, groups[50])

# The classifier for 1000 classes
outputs = classifier(x, 1000)

# Instantiate the Model
model = Model(inputs, outputs)

model.summary()
