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


# MobileNet
from keras import layers, Input, Model

def stem(inputs, alpha):
    """ Create the stem group
        inputs : input tensor
        alpha  : width multiplier
    """
    # Apply the width filter to the number of feature maps
    filters = int(32 * alpha)

    # Normal Convolutional block
    x = layers.ZeroPadding2D(padding=((0, 1), (0, 1)))(inputs)
    x = layers.Conv2D(32, (3, 3), strides=(2, 2), padding='valid')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Depthwise Separable Convolution Block
    x = depthwise_block(x, 64, alpha, (1, 1))
    return x

def classifier(x, alpha, dropout, nb_classes):
    """ Create the classifier group
        inputs    : input tensor
        alpha     : width multiplier
        dropout   : dropout percentage
        nb_classes: number of output classes
    """

    # Flatten the feature maps into 1D feature maps (?, N)
    x = layers.GlobalAveragePooling2D()(x)

    # Reshape the feature maps to (?, 1, 1, 1024)
    shape = (1, 1, int(1024 * alpha))
    x = layers.Reshape(shape)(x)
    # Perform dropout for preventing overfitting
    x = layers.Dropout(dropout)(x)

    # Use convolution for classifying (emulates a fully connected layer)
    x = layers.Conv2D(nb_classes, (1, 1), padding='same')(x)
    x = layers.Activation('softmax')(x)
    # Reshape the resulting output to 1D vector of number of classes
    x = layers.Reshape((nb_classes, ))(x)

    return x

def depthwise_block(x, nb_filters, alpha, strides):
    """ Create a Depthwise Separable Convolution block
        inputs    : input tensor
        nb_filters: number of filters
        alpha     : width multiplier
        strides   : strides
    """
    # Apply the width filter to the number of feature maps
    filters = int(nb_filters * alpha)

    # Strided convolution to match number of filters
    if strides == (2, 2):
        x = layers.ZeroPadding2D(padding=((0, 1), (0, 1)))(x)
        padding = 'valid'
    else:
        padding = 'same'

    # Depthwise Convolution
    x = layers.DepthwiseConv2D((3, 3), strides, padding=padding)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Pointwise Convolution
    x = layers.Conv2D(filters, (1, 1), strides=(1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


alpha      = 1    # width multiplier
dropout    = 0.5  # dropout percentage
nb_classes = 1000 # number of classes

inputs = Input(shape=(224, 224, 3))

# Create the stem group
x = stem(inputs, alpha)

# First Depthwise Separable Convolution Group
# Strided convolution - feature map size reduction
x = depthwise_block(x, 128, alpha, strides=(2, 2))
x = depthwise_block(x, 128, alpha, strides=(1, 1))

# Second Depthwise Separable Convolution Group
# Strided convolution - feature map size reduction
x = depthwise_block(x, 256, alpha, strides=(2, 2))
x = depthwise_block(x, 256, alpha, strides=(1, 1))

# Third Depthwise Separable Convolution Group
# Strided convolution - feature map size reduction
x = depthwise_block(x, 512, alpha, strides=(2, 2))
for _ in range(5):
    x = depthwise_block(x, 512, alpha, strides=(1, 1))

# Fourth Depthwise Separable Convolution Group
# Strided convolution - feature map size reduction
x = depthwise_block(x, 1024, alpha, strides=(2, 2))
x = depthwise_block(x, 1024, alpha, strides=(1, 1))

x = classifier(x, alpha, dropout, nb_classes)

print(x)

model = Model(inputs, x)
model.summary()

