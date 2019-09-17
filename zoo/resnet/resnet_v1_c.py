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

# ResNet (50, 101, 152 + Composable) v1.0
# Paper: https://arxiv.org/pdf/1512.03385.pdf

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, ReLU, BatchNormalization, ZeroPadding2D
from tensorflow.keras.layers import MaxPooling2D, Dense, Add, GlobalAveragePooling2D

class ResNetV1(object):
    """ Residual Convolutional Neural Network V1
    """
    # Meta-parameter: list of groups: filter size and number of blocks
    groups = { 50 : [ (64, 3), (128, 4), (256, 6),  (512, 3) ],		# ResNet50
               101: [ (64, 3), (128, 4), (256, 23), (512, 3) ],		# ResNet101
               152: [ (64, 3), (128, 8), (256, 36), (512, 3) ]		# ResNet152
             }
    init_weights='he_normal'
    _model = None
    
    def __init__(self, n_layers, input_shape=(224, 224, 3), n_classes=1000):
        """ Construct a Residual Convolutional Neural Network V1
	    n_layers   : number of layers
	    input_shape: input shape
    	    n_classes  : number of output classes
        """
        if n_layers not in [50, 101, 152]:
            raise Exception("ResNet: Invalid value for n_layers")
        
        # The input tensor
        inputs = Input(input_shape)

        # The stem convolutional group
        x = self.stem(inputs)

        # The learner
        x = self.learner(x, self.groups[n_layers])

        # The classifier for 1000 classes
        outputs = self.classifier(x, n_classes)

        # Instantiate the Model
        self._model = Model(inputs, outputs)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, _model):
        self._model = _model
        
    def stem(self, inputs):
        """ Construct the Stem Convolutional Group 
            inputs : the input vector
        """
        # The 224x224 images are zero padded (black - no signal) to be 230x230 images prior to the first convolution
        x = ZeroPadding2D(padding=(3, 3))(inputs)
    
        # First Convolutional layer which uses a large (coarse) filter 
        x = Conv2D(64, (7, 7), strides=(2, 2), padding='valid', use_bias=False, kernel_initializer=self.init_weights)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
    
        # Pooled feature maps will be reduced by 75%
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)
        return x
    
    def learner(self, x, groups):
        """ Construct the Learner
            x     : input to the learner
            groups: list of groups: number of filters and blocks
        """
        # First Residual Block Group (not strided)
        n_filters, n_blocks = groups.pop(0)
        x = self.group(x, n_filters, n_blocks, strides=(1, 1))

        # Remaining Residual Block Groups (strided)
        for n_filters, n_blocks in groups:
            x = ResNetV1.group(x, n_filters, n_blocks)
        return x

    @staticmethod
    def group(x, n_filters, n_blocks, strides=(2, 2), init_weights=None):
        """ Construct a Residual Group 
            x         : input into the group
            n_filters : number of filters for the group
            n_blocks  : number of residual blocks with identity link
            strides   : whether the projection block is a strided convolution
        """
        # Double the size of filters to fit the first Residual Group
        x = ResNetV1.projection_block(x, n_filters, strides=strides, init_weights=init_weights)

        # Identity residual blocks
        for _ in range(n_blocks):
            x = ResNetV1.identity_block(x, n_filters, init_weights)
        return x

    @staticmethod
    def identity_block(x, n_filters, init_weights=None):
        """ Construct a Bottleneck Residual Block with Identity Link
            x        : input into the block
            n_filters: number of filters
        """
        if init_weights is None:
            init_weights = ResNetV1.init_weights
            
        # Save input vector (feature maps) for the identity link
        shortcut = x
    
        ## Construct the 1x1, 3x3, 1x1 residual block (fig 3c)

        # Dimensionality reduction
        x = Conv2D(n_filters, (1, 1), strides=(1, 1), use_bias=False, kernel_initializer=init_weights)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        # Bottleneck layer
        x = Conv2D(n_filters, (3, 3), strides=(1, 1), padding="same", use_bias=False, kernel_initializer=init_weights)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        # Dimensionality restoration - increase the number of output filters by 4X
        x = Conv2D(n_filters * 4, (1, 1), strides=(1, 1), use_bias=False, kernel_initializer=init_weights)(x)
        x = BatchNormalization()(x)

        # Add the identity link (input) to the output of the residual block
        x = Add()([shortcut, x])
        x = ReLU()(x)
        return x

    @staticmethod
    def projection_block(x, n_filters, strides=(2,2), init_weights=None):
        """ Construct Bottleneck Residual Block with Projection Shortcut
            Increase the number of filters by 4X
            x        : input into the block
            n_filters: number of filters
            strides  : whether entry convolution is strided (i.e., (2, 2) vs (1, 1))
        """
        if init_weights is None:
            init_weights = ResNetV1.init_weights
            
        # Construct the projection shortcut
        # Increase filters by 4X to match shape when added to output of block
        shortcut = Conv2D(4 * n_filters, (1, 1), strides=strides, use_bias=False, kernel_initializer=init_weights)(x)
        shortcut = BatchNormalization()(shortcut)

        ## Construct the 1x1, 3x3, 1x1 residual block (fig 3c)

        # Dimensionality reduction
        # Feature pooling when strides=(2, 2)
        x = Conv2D(n_filters, (1, 1), strides=strides, use_bias=False, kernel_initializer=init_weights)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        # Bottleneck layer
        x = Conv2D(n_filters, (3, 3), strides=(1, 1), padding='same', use_bias=False, kernel_initializer=init_weights)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        # Dimensionality restoration - increase the number of filters by 4X
        x = Conv2D(4 * n_filters, (1, 1), strides=(1, 1), use_bias=False, kernel_initializer=init_weights)(x)
        x = BatchNormalization()(x)

        # Add the projection shortcut link to the output of the residual block
        x = Add()([x, shortcut])
        x = ReLU()(x)
        return x

    def classifier(self, x, n_classes):
      """ Construct the Classifier Group 
          x         : input to the classifier
          n_classes : number of output classes
      """
      # Pool at the end of all the convolutional residual blocks
      x = GlobalAveragePooling2D()(x)

      # Final Dense Outputting Layer for the outputs
      outputs = Dense(n_classes, activation='softmax', kernel_initializer=self.init_weights)(x)
      return outputs

# Example of ResNet50
# resnet = ResNetV1(50)
