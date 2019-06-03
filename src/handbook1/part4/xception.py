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

# Xception
from keras import layers, Input, Model

def entryFlow(inputs):
    """ Create the entry flow section
        inputs : input tensor to neural network
    """

    def stem(inputs):
        """ Create the stem entry into the neural network
            inputs : input tensor to neural network
        """
        # First convolution
        x = layers.Conv2D(32, (3, 3), strides=(2, 2))(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        # Second convolution, double the number of filters
        x = layers.Conv2D(64, (3, 3), strides=(1, 1))(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        return x

    # Create the stem to the neural network
    x = stem(inputs)

    # Create three residual blocks
    for nb_filters in [128, 256, 728]:
        x = residual_block_entry(x, nb_filters)

    return x

def middleFlow(x):
    """ Create the middle flow section
        x : input tensor into section
    """

    # Create 8 residual blocks
    for _ in range(8):
        x = residual_block_middle(x, 728)
    return x

def exitFlow(x):
    """ Create the exit flow section
        x : input tensor into section
    """
    def classifier(x):
        """ The output classifier
            x : input tensor
        """
        # Global Average Pooling will flatten the 10x10 feature maps into 1D
        # feature maps
        x = layers.GlobalAveragePooling2D()(x)
        # Fully connected output layer (classification)
        x = layers.Dense(1000, activation='softmax')(x)
        return x

    shortcut = x

    # First Depthwise Separable Convolution
    x = layers.SeparableConv2D(728, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)

    # Second Depthwise Separable Convolution
    x = layers.SeparableConv2D(1024, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Create pooled feature maps, reduce size by 75%
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Add strided convolution to identity link to double number of filters to
    # match output of residual block for the add operation
    shortcut = layers.Conv2D(1024, (1, 1), strides=(2, 2),
                             padding='same')(shortcut)
    shortcut = layers.BatchNormalization()(shortcut)

    x = layers.add([x, shortcut])

    # Third Depthwise Separable Convolution
    x = layers.SeparableConv2D(1556, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Fourth Depthwise Separable Convolution
    x = layers.SeparableConv2D(2048, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Create classifier section
    x = classifier(x)

    return x

def residual_block_entry(x, nb_filters):
    """ Create a residual block using Depthwise Separable Convolutions
        x         : input into residual block
        nb_filters: number of filters
    """
    shortcut = x

    # First Depthwise Separable Convolution
    x = layers.SeparableConv2D(nb_filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Second depthwise Separable Convolution
    x = layers.SeparableConv2D(nb_filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Create pooled feature maps, reduce size by 75%
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Add strided convolution to identity link to double number of filters to
    # match output of residual block for the add operation
    shortcut = layers.Conv2D(nb_filters, (1, 1), strides=(2, 2),
                             padding='same')(shortcut)
    shortcut = layers.BatchNormalization()(shortcut)

    x = layers.add([x, shortcut])

    return x

def residual_block_middle(x, nb_filters):
    """ Create a residual block using Depthwise Separable Convolutions
        x         : input into residual block
        nb_filters: number of filters
    """
    shortcut = x

    # First Depthwise Separable Convolution
    x = layers.SeparableConv2D(nb_filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Second depthwise Separable Convolution
    x = layers.SeparableConv2D(nb_filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Third depthwise Separable Convolution
    x = layers.SeparableConv2D(nb_filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.add([x, shortcut])
    return x

inputs = Input(shape=(299, 299, 3))

# Create entry section
x = entryFlow(inputs)
# Create the middle section
x = middleFlow(x)
# Create the exit section
x = exitFlow(x)

model = Model(inputs, x)
model.summary()

