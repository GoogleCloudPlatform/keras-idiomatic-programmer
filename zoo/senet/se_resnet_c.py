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

# SE-ResNet (50/101/152 + composable) v1.0
# Paper: https://arxiv.org/pdf/1709.01507.pdf

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import ZeroPadding2D, Conv2D, MaxPooling2D, BatchNormalization, ReLU
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Reshape, Multiply, Add

class SEResNet(object):
    """ Construct a Squeeze & Excite Residual Convolution Neural Network """
    # Meta-parameter: list of groups: filter size and number of blocks
    groups = { 50 : [ (64, 3), (128, 4), (256, 6),  (512, 3) ],         # SE-ResNet50
               101: [ (64, 3), (128, 4), (256, 23), (512, 3) ],         # SE-ResNet101
               152: [ (64, 3), (128, 8), (256, 36), (512, 3) ]          # SE-ResNet152
             }
    # Meta-parameter: Amount of filter reduction in squeeze operation
    init_weights = 'he_normal'
    _model = None

    def __init__(self, n_layers, ratio=16, input_shape=(224, 224, 3), n_classes=1000):
        """ Construct a Residual Convolutional Neural Network V1
            n_layers   : number of layers
            input_shape: input shape
            n_classes  : number of output classes
        """
        if n_layers not in [50, 101, 152]:
            raise Exception("ResNet: Invalid value for n_layers")

        # The input tensor
        inputs = Input(shape=input_shape)

        # The Stem Group
        x = self.stem(inputs)

        # The Learner
        x = self.learner(x, self.groups[n_layers], ratio)

        # The Classifier for 1000 classes
        outputs = self.classifier(x, 1000)

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

    def learner(self, x, groups, ratio):
        """ Construct the Learner
            x     : input to the learner
            groups: list of groups: number of filters and blocks
            ratio : amount of filter reduction in squeeze
        """
        # First Residual Block Group (not strided)
        n_filters, n_blocks = groups.pop(0)
        x = SEResNet.group(x, n_filters, n_blocks, ratio, strides=(1, 1))

        # Remaining Residual Block Groups (strided)
        for n_filters, n_blocks in groups:
            x = SEResNet.group(x, n_filters, n_blocks, ratio)
        return x	

    @staticmethod
    def group(x, n_filters, n_blocks, ratio, strides=(2, 2), init_weights=None):
        """ Construct the Squeeze-Excite Group
            x        : input to the group
            n_blocks : number of blocks
            n_filters: number of filters
            ratio    : amount of filter reduction during squeeze
            strides  : whether projection block is strided
        """
        # first block uses linear projection to match the doubling of filters between groups
        x = SEResNet.projection_block(x, n_filters, strides=strides, ratio=ratio, init_weights=init_weights)

        # remaining blocks use identity link
        for _ in range(n_blocks-1):
            x = SEResNet.identity_block(x, n_filters, ratio=ratio, init_weights=init_weights)
        return x

    @staticmethod
    def squeeze_excite_block(x, ratio=16, init_weights=None):
        """ Create a Squeeze and Excite block
            x     : input to the block
            ratio : amount of filter reduction during squeeze
        """  
        if init_weights is None:
            init_weights = SEResNet.init_weights
            
        # Remember the input
        shortcut = x
    
        # Get the number of filters on the input
        filters = x.shape[-1]

        # Squeeze (dimensionality reduction)
        # Do global average pooling across the filters, which will the output a 1D vector
        x = GlobalAveragePooling2D()(x)
    
        # Reshape into 1x1 feature maps (1x1xC)
        x = Reshape((1, 1, filters))(x)
    
        # Reduce the number of filters (1x1xC/r)
        x = Dense(filters // ratio, activation='relu', kernel_initializer=init_weights, use_bias=False)(x)

        # Excitation (dimensionality restoration)
        # Restore the number of filters (1x1xC)
        x = Dense(filters, activation='sigmoid', kernel_initializer=init_weights, use_bias=False)(x)

        # Scale - multiply the squeeze/excitation output with the input (WxHxC)
        x = Multiply()([shortcut, x])
        return x

    @staticmethod
    def identity_block(x, n_filters, ratio=16, init_weights=None):
        """ Create a Bottleneck Residual Block with Identity Link
            x        : input into the block
            n_filters: number of filters
            ratio    : amount of filter reduction during squeeze
        """
        if init_weights is None:
            init_weights = SEResNet.init_weights

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
    
        # Pass the output through the squeeze and excitation block
        x = SEResNet.squeeze_excite_block(x, ratio, init_weights)
    
        # Add the identity link (input) to the output of the residual block
        x = Add()([shortcut, x])
        x = ReLU()(x)
        return x

    @staticmethod
    def projection_block(x, n_filters, strides=(2,2), ratio=16, init_weights=None):
        """ Create Bottleneck Residual Block with Projection Shortcut
            Increase the number of filters by 4X
            x        : input into the block
            n_filters: number of filters
            strides  : whether entry convolution is strided (i.e., (2, 2) vs (1, 1))
            ratio    : amount of filter reduction during squeeze
        """
        if init_weights is None:
            init_weights = SEResNet.init_weights
            
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

        # Pass the output through the squeeze and excitation block
        x = SEResNet.squeeze_excite_block(x, ratio, init_weights)

        # Add the projection shortcut link to the output of the residual block
        x = Add()([x, shortcut])
        x = ReLU()(x)
        return x

    def classifier(self, x, n_classes):
      """ Create the Classifier Group 
          x         : input to the classifier
          n_classes : number of output classes
      """
      # Pool at the end of all the convolutional residual blocks
      x = GlobalAveragePooling2D()(x)

      # Final Dense Outputting Layer for the outputs
      outputs = Dense(n_classes, activation='softmax', kernel_initializer=self.init_weights)(x)
      return outputs

# Example
# senet = SEResNet(50)


