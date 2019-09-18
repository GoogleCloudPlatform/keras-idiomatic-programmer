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


# MobileNet v2 + composable (2019)
# Paper: https://arxiv.org/pdf/1801.04381.pdf
# 224x224 input: 3,504,872 parameters

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import ZeroPadding2D, Conv2D, BatchNormalization, ReLU
from tensorflow.keras.layers import DepthwiseConv2D, Add, GlobalAveragePooling2D, Dense

class MobileNetV2(object):
    """ Construct a Mobile Convolution Neural Network """
    # Meta-parameter: width multiplier (0 .. 1) for reducing number of filters.
    alpha = 1
    # Meta-parameter: multiplier to expand the number of filters
    expansion = 6
    init_weights = 'glorot_uniform'
    _model = None

    def __init__(self, alpha=1, expansion=6, input_shape=(224, 224, 3), n_classes=1000):
        """ Construct a Mobile Convolution Neural Network
            alpha      : width multiplier
            expansion  : multiplier to expand the number of filters
            input_shape: the input shape
            n_classes  : number of output classes
        """
        inputs = Input(shape=(224, 224, 3))

        # The Stem Group
        x = self.stem(inputs, alpha)    

        # The Learner
        x = self.learner(x, alpha, expansion)

        # The classifier 
        outputs = self.classifier(x, n_classes)

        # Instantiate the Model
        self._model = Model(inputs, outputs)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, _model):
        self._model = model

    def stem(self, inputs, alpha):
        """ Construct the Stem Group
            inputs : input tensor
            alpha  : width multiplier
        """
        # Calculate the number of filters for the stem convolution
        # Must be divisible by 8
        n_filters = max(8, (int(32 * alpha) + 4) // 8 * 8)
    
        # Convolutional block
        x = ZeroPadding2D(padding=((0, 1), (0, 1)))(inputs)
        x = Conv2D(n_filters, (3, 3), strides=(2, 2), padding='valid', use_bias=False, kernel_initializer=self.init_weights)(x)
        x = BatchNormalization()(x)
        x = ReLU(6.)(x)
        return x
    
    def learner(self, x, alpha, expansion=6):
        """ Construct the Learner
            x        : input to the learner
            alpha    : width multiplier
            expansion: multipler to expand number of filters
        """
        # First Inverted Residual Convolution Group
        x = MobileNetV2.group(x, 16, 1, alpha, expansion=1, strides=(1, 1))
    
        # Second Inverted Residual Convolution Group
        x = MobileNetV2.group(x, 24, 2, alpha, expansion)

        # Third Inverted Residual Convolution Group
        x = MobileNetV2.group(x, 32, 3, alpha, expansion)
    
        # Fourth Inverted Residual Convolution Group
        x = MobileNetV2.group(x, 64, 4, alpha, expansion)
    
        # Fifth Inverted Residual Convolution Group
        x = MobileNetV2.group(x, 96, 3, alpha, expansion, strides=(1, 1))
    
        # Sixth Inverted Residual Convolution Group
        x = MobileNetV2.group(x, 160, 3, alpha, expansion)
    
        # Seventh Inverted Residual Convolution Group
        x = MobileNetV2.group(x, 320, 1, alpha, expansion, strides=(1, 1))
    
        # Last block is a 1x1 linear convolutional layer,
        # expanding the number of filters to 1280.
        x = Conv2D(1280, (1, 1), use_bias=False, kernel_initializer=self.init_weights)(x)
        x = BatchNormalization()(x)
        x = ReLU(6.)(x)
        return x

    @staticmethod
    def group(x, n_filters, n_blocks, alpha, expansion=6, strides=(2, 2)):
        """ Construct an Inverted Residual Group
            x         : input to the group
            n_filters : number of filters
            n_blocks  : number of blocks in the group
            alpha     : width multiplier
            expansion : multiplier for expanding the number of filters
            strides   : whether first inverted residual block is strided.
        """   
        # In first block, the inverted residual block maybe strided - feature map size reduction
        x = MobileNetV2.inverted_block(x, n_filters, alpha, expansion, strides=strides)
    
        # Remaining blocks
        for _ in range(n_blocks - 1):
            x = MobileNetV2.inverted_block(x, n_filters, alpha, expansion, strides=(1, 1))
        return x

    @staticmethod
    def inverted_block(x, n_filters, alpha, expansion=6, strides=(1, 1), init_weights=None):
        """ Construct an Inverted Residual Block
            x         : input to the block
            n_filters : number of filters
            alpha     : width multiplier
            strides   : strides
            expansion : multiplier for expanding number of filters
        """
        if init_weights is None:
            init_weights = MobileNetV2.init_weights
            
        # Remember input
        shortcut = x

        # Apply the width filter to the number of feature maps for the pointwise convolution
        filters = int(n_filters * alpha)
    
        n_channels = int(x.shape[3])
    
        # Dimensionality Expansion (non-first block)
        if expansion > 1:
            # 1x1 linear convolution
            x = Conv2D(expansion * n_channels, (1, 1), padding='same', use_bias=False, kernel_initializer=init_weights)(x)
            x = BatchNormalization()(x)
            x = ReLU(6.)(x)

        # Strided convolution to match number of filters
        if strides == (2, 2):
            x = ZeroPadding2D(padding=((0, 1), (0, 1)))(x)
            padding = 'valid'
        else:
            padding = 'same'

        # Depthwise Convolution
        x = DepthwiseConv2D((3, 3), strides, padding=padding, use_bias=False, kernel_initializer=init_weights)(x)
        x = BatchNormalization()(x)
        x = ReLU(6.)(x)

        # Linear Pointwise Convolution
        x = Conv2D(filters, (1, 1), strides=(1, 1), padding='same', use_bias=False, kernel_initializer=init_weights)(x)
        x = BatchNormalization()(x)
    
        # Number of input filters matches the number of output filters
        if n_channels == filters and strides == (1, 1):
            x = Add()([shortcut, x]) 
        return x

    def classifier(self, x, n_classes):
        """ Construct the classifier group
            x         : input to the classifier
            n_classes : number of output classes
        """
        # Flatten the feature maps into 1D feature maps (?, N)
        x = GlobalAveragePooling2D()(x)

        # Dense layer for final classification
        x = Dense(n_classes, activation='softmax', kernel_initializer=self.init_weights)(x)
        return x

# Example
# mobilenet = MobileNetV2()


