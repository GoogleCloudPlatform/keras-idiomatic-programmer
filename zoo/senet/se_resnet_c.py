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
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Reshape, Multiply, Add, Activation
from tensorflow.keras.regularizers import l2

import sys
sys.path.append('../')
from models_c import Composable

class SEResNet(Composable):
    """ Construct a Squeeze & Excite Residual Convolution Neural Network """
    # Meta-parameter: list of groups: filter size and number of blocks
    groups = { 50 : [ { 'n_filters' : 64, 'n_blocks': 3 },
                      { 'n_filters': 128, 'n_blocks': 4 },
                      { 'n_filters': 256, 'n_blocks': 6 },
                      { 'n_filters': 512, 'n_blocks': 3 } ],            # SE-ResNet50
               101: [ { 'n_filters' : 64, 'n_blocks': 3 },
                      { 'n_filters': 128, 'n_blocks': 4 },
                      { 'n_filters': 256, 'n_blocks': 23 },
                      { 'n_filters': 512, 'n_blocks': 3 } ],            # SE-ResNet101
               152: [ { 'n_filters' : 64, 'n_blocks': 3 },
                      { 'n_filters': 128, 'n_blocks': 8 },
                      { 'n_filters': 256, 'n_blocks': 36 },
                      { 'n_filters': 512, 'n_blocks': 3 } ]             # SE-ResNet152
	      }
    # Meta-parameter: Amount of filter reduction in squeeze operation
    ratio = 16
    
    def __init__(self, n_layers, ratio=16, input_shape=(224, 224, 3), n_classes=1000,
                 reg=l2(0.001), init_weights='he_normal', relu=None):
        """ Construct a Residual Convolutional Neural Network V1
            n_layers    : number of layers
            input_shape : input shape
            n_classes   : number of output classes
            reg         : kernel regularizer
            init_weights: kernel initializer
            relu        : max value for ReLU
        """
        super().__init__(reg=reg, init_weights=init_weights, relu=relu)
        
        # predefined
        if isinstance(n_layers, int):
            if n_layers not in [50, 101, 152]:
                raise Exception("SE-ResNet: Invalid value for n_layers")
            groups = list(self.groups[n_layers])
        # user defined
        else:
            groups = n_layers

        # The input tensor
        inputs = Input(shape=input_shape)

        # The Stem Group
        x = self.stem(inputs)

        # The Learner
        x = self.learner(x, groups=groups, ratio=ratio)

        # The Classifier 
        outputs = self.classifier(x, n_classes)

        # Instantiate the Model
        self._model = Model(inputs, outputs)
    
    def stem(self, inputs):
        """ Construct the Stem Convolutional Group 
            inputs : the input vector
        """
        # The 224x224 images are zero padded (black - no signal) to be 230x230 images prior to the first convolution
        x = ZeroPadding2D(padding=(3, 3))(inputs)
    
        # First Convolutional layer which uses a large (coarse) filter 
        x = self.Conv2D(x, 64, (7, 7), strides=(2, 2), padding='valid', use_bias=False)
        x = BatchNormalization()(x)
        x = self.ReLU(x)
    
        # Pooled feature maps will be reduced by 75%
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)
        return x

    def learner(self, x, **metaparameters):
        """ Construct the Learner
            x     : input to the learner
            groups: list of groups: number of filters and blocks
        """
        groups = metaparameters['groups']

        # First Residual Block Group (not strided)
        x = SEResNet.group(x, strides=(1, 1), **groups.pop(0), **metaparameters)

        # Remaining Residual Block Groups (strided)
        for group in groups:
            x = SEResNet.group(x, **group, **metaparameters)
        return x	

    @staticmethod
    def group(x, strides=(2, 2), **metaparameters):
        """ Construct the Squeeze-Excite Group
            x        : input to the group
            strides  : whether projection block is strided
            n_blocks : number of blocks
        """
        n_blocks  = metaparameters['n_blocks']

        # first block uses linear projection to match the doubling of filters between groups
        x = SEResNet.projection_block(x, strides=strides, **metaparameters)

        # remaining blocks use identity link
        for _ in range(n_blocks-1):
            x = SEResNet.identity_block(x, **metaparameters)
        return x

    @staticmethod
    def squeeze_excite_block(x, **metaparameters):
        """ Create a Squeeze and Excite block
            x     : input to the block
            ratio : amount of filter reduction during squeeze
        """  
        if 'ratio' in metaparameters:
            ratio = metaparameters['ratio']
        else:
            ratio = SEResNet.ratio
            
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
        x = self.Dense(x, filters // ratio, activation='relu', use_bias=False, **metaparameters)

        # Excitation (dimensionality restoration)
        # Restore the number of filters (1x1xC)
        x = self.Dense(x, filters, activation='sigmoid', use_bias=False, **metaparameters)

        # Scale - multiply the squeeze/excitation output with the input (WxHxC)
        x = Multiply()([shortcut, x])
        return x

    @staticmethod
    def identity_block(x, **metaparameters):
        """ Create a Bottleneck Residual Block with Identity Link
            x        : input into the block
            n_filters: number of filters
        """
        n_filters = metaparameters['n_filters']
        del metaparameters['n_filters']

        # Save input vector (feature maps) for the identity link
        shortcut = x
    
        ## Construct the 1x1, 3x3, 1x1 residual block (fig 3c)

        # Dimensionality reduction
        x = Composable.Conv2D(x, n_filters, (1, 1), strides=(1, 1), use_bias=False, 
                              **metaparameters)
        x = BatchNormalization()(x)
        x = Composable.ReLU(x)

        # Bottleneck layer
        x = Composable.Conv2D(x, n_filters, (3, 3), strides=(1, 1), padding="same", use_bias=False, 
                              **metaparameters)
        x = BatchNormalization()(x)
        x = Composable.ReLU(x)

        # Dimensionality restoration - increase the number of output filters by 4X
        x = Composable.Conv2D(x, n_filters * 4, (1, 1), strides=(1, 1), use_bias=False, 
                              **metaparameters)
        x = BatchNormalization()(x)
    
        # Pass the output through the squeeze and excitation block
        x = SEResNet.squeeze_excite_block(x, **metaparameters)
    
        # Add the identity link (input) to the output of the residual block
        x = Add()([shortcut, x])
        x = Composable.ReLU(x)
        return x

    @staticmethod
    def projection_block(x, strides=(2,2), **metaparameters):
        """ Create Bottleneck Residual Block with Projection Shortcut
            Increase the number of filters by 4X
            x        : input into the block
            strides  : whether entry convolution is strided (i.e., (2, 2) vs (1, 1))
            n_filters: number of filters
        """
        n_filters = metaparameters['n_filters']
        del metaparameters['n_filters']
            
        # Construct the projection shortcut
        # Increase filters by 4X to match shape when added to output of block
        shortcut = Composable.Conv2D(4 * n_filters, (1, 1), strides=strides, use_bias=False, 
                                     **metaparameters)
        shortcut = BatchNormalization()(shortcut)

        ## Construct the 1x1, 3x3, 1x1 residual block (fig 3c)

        # Dimensionality reduction
        # Feature pooling when strides=(2, 2)
        x = Composable.Conv2D(x, n_filters, (1, 1), strides=strides, use_bias=False, 
                              **metaparameters)
        x = BatchNormalization()(x)
        x = Composable.ReLU(x)

        # Bottleneck layer
        x = Composable.Conv2D(x, n_filters, (3, 3), strides=(1, 1), padding='same', use_bias=False, 
                              **metaparameters)
        x = BatchNormalization()(x)
        x = Composable.ReLU(x)

        # Dimensionality restoration - increase the number of filters by 4X
        x = Composable.Conv2D(x, 4 * n_filters, (1, 1), strides=(1, 1), use_bias=False, 
                              **metaparameters)
        x = BatchNormalization()(x)

        # Pass the output through the squeeze and excitation block
        x = SEResNet.squeeze_excite_block(x, **metaparameters)

        # Add the projection shortcut link to the output of the residual block
        x = Add()([x, shortcut])
        x = Composable.ReLU(x)
        return x

    def classifier(self, x, n_classes):
      """ Create the Classifier Group 
          x         : input to the classifier
          n_classes : number of output classes
      """
      # Save the encoding layer
      self.encoding = x
      
      # Pool at the end of all the convolutional residual blocks
      x = GlobalAveragePooling2D()(x)

      # Save the embedding layer
      self.embedding = x

      # Final Dense Outputting Layer for the outputs
      x = self.Dense(x, n_classes)
      # Save the pre-activation probabilities layer
      self.probabilities = x
      outputs = Activation('softmax')(x)
      # Save the post-activation probabilities layer
      self.softmax = outputs
      return outputs

# Example
# senet = SEResNet(50)
