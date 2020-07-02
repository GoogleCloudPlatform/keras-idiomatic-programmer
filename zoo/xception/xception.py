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

# Xception (2016)
# https://arxiv.org/pdf/1610.02357.pdf

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Dense, GlobalAveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, MaxPooling2D, Add

def entryFlow(inputs):
    """ Create the entry flow section
        inputs : input tensor to neural network
    """

    def stem(inputs):
        """ Create the stem entry into the neural network
            inputs : input tensor to neural network
        """
        # Strided convolution - dimensionality reduction
        # Reduce feature maps by 75%
        x = Conv2D(32, (3, 3), strides=(2, 2))(inputs)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        # Convolution - dimensionality expansion
        # Double the number of filters
        x = Conv2D(64, (3, 3), strides=(1, 1))(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        return x

    # Create the stem to the neural network
    x = stem(inputs)

    # Create three residual blocks using linear projection
    for n_filters in [128, 256, 728]:
        x = projection_block(x, n_filters)

    return x

def middleFlow(x):
    """ Create the middle flow section
        x : input tensor into section
    """
    # Create 8 residual blocks
    for _ in range(8):
        x = residual_block(x, 728)
    return x

def exitFlow(x, n_classes):
    """ Create the exit flow section
        x         : input to the exit flow section
        n_classes : number of output classes
    """
    def classifier(x, n_classes):
        """ The output classifier
            x         : input to the classifier
            n_classes : number of output classes
        """
        # Global Average Pooling will flatten the 10x10 feature maps into 1D
        # feature maps
        x = GlobalAveragePooling2D()(x)
        
        # Fully connected output layer (classification)
        x = Dense(n_classes, activation='softmax')(x)
        return x

    # 1x1 strided convolution to increase number and reduce size of feature maps
    # in identity link to match output of residual block for the add operation (projection shortcut)
    shortcut = Conv2D(1024, (1, 1), strides=(2, 2), padding='same')(x)
    shortcut = BatchNormalization()(shortcut)

    # First Depthwise Separable Convolution
    # Dimensionality reduction - reduce number of filters
    x = SeparableConv2D(728, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Second Depthwise Separable Convolution
    # Dimensionality restoration
    x = SeparableConv2D(1024, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Create pooled feature maps, reduce size by 75%
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Add the projection shortcut to the output of the pooling layer
    x = Add()([x, shortcut])

    # Third Depthwise Separable Convolution
    x = SeparableConv2D(1556, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Fourth Depthwise Separable Convolution
    x = SeparableConv2D(2048, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Create classifier section
    x = classifier(x, n_classes)

    return x

def projection_block(x, n_filters):
    """ Create a residual block using Depthwise Separable Convolutions with Projection shortcut
        x        : input into residual block
        n_filters: number of filters
    """
    # Remember the input
    shortcut = x
    
    # Strided convolution to double number of filters in identity link to
    # match output of residual block for the add operation (projection shortcut)
    shortcut = Conv2D(n_filters, (1, 1), strides=(2, 2), padding='same')(shortcut)
    shortcut = BatchNormalization()(shortcut)

    # First Depthwise Separable Convolution
    x = SeparableConv2D(n_filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Second depthwise Separable Convolution
    x = SeparableConv2D(n_filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Create pooled feature maps, reduce size by 75%
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Add the projection shortcut to the output of the block
    x = Add()([x, shortcut])

    return x

def residual_block(x, n_filters):
    """ Create a residual block using Depthwise Separable Convolutions
        x        : input into residual block
        n_filters: number of filters
    """
    # Remember the input
    shortcut = x

    # First Depthwise Separable Convolution
    x = SeparableConv2D(n_filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Second depthwise Separable Convolution
    x = SeparableConv2D(n_filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Third depthwise Separable Convolution
    x = SeparableConv2D(n_filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # Add the identity link to the output of the block
    x = Add()([x, shortcut])
    return x

# Create the input vector
inputs = Input(shape=(299, 299, 3))

# Create entry section
x = entryFlow(inputs)

# Create the middle section
x = middleFlow(x)

# Create the exit section for 1000 classes
outputs = exitFlow(x, 1000)

# Instantiate the model
model = Model(inputs, outputs)
