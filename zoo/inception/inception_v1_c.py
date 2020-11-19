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

# Inception v1 (GoogLeNet) - Composable
# Trainable params: 12,997,352
# Paper: https://arxiv.org/pdf/1409.4842.pdf

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, ReLU, ZeroPadding2D, Flatten, Dropout, Activation
from tensorflow.keras.layers import MaxPooling2D, Dense, Concatenate, AveragePooling2D

import sys
sys.path.append('../')
from models_c import Composable

class InceptionV1(Composable):
    """ Construct an Inception Convolutional Neural Network """

    # Initial Hyperparameters
    hyperparameters = { 'initializer': 'glorot_uniform',
                        'regularizer': None,
                        'relu_clip'  : None,
                        'bn_epsilon' : None,
                        'use_bias'   : True
                      }

    def __init__(self, dropout=0.4, 
                 input_shape=(224, 224, 3), n_classes=1000, include_top=True,
                 **hyperparameters):
        """ Construct an Inception Convolutional Neural Network
            dropout     : percentage of dropout
            input_shape : input shape to the neural network
            n_classes   : number of output classes
            include_top : whether to include the classifier
            initializer : kernel initializer
            regularizer : kernel regularizer
            relu_clip   : max value for ReLU
            bn_epsilon  : epsilon for batch norm
            use_bias    : whether to use bias
        """
        # Configure base (super) class
        Composable.__init__(self, input_shape, include_top, self.hyperparameters, **hyperparameters)
        
	# The input tensor
        inputs = Input(shape=input_shape)

        # The stem convolutional group
        x = self.stem(inputs)

        # The learner
        outputs, aux = self.learner(x, n_classes)

        # The classifier 
        if include_top:
            outputs = self.classifier(outputs, n_classes, dropout)

        # Instantiate the Model
        self._model = Model(inputs, [outputs] + aux)

    def stem(self, inputs):
        """ Construct the Stem Convolutional Group 
            inputs : the input vector
        """
        # The 224x224 images are zero padded (black - no signal) to be 230x230 images prior to the first convolution
        x = ZeroPadding2D(padding=(3, 3))(inputs)
    
        # First Convolutional layer which uses a large (coarse) filter 
        x = self.Conv2D(x, 64, (7, 7), strides=(2, 2), padding='valid')
        x = self.ReLU(x)

        # Pooled feature maps will be reduced by 75%
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        # Second Convolutional layer which uses a mid-size filter
        x = self.Conv2D(x, 64, (1, 1), strides=(1, 1), padding='same')
        x = self.ReLU(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = self.Conv2D(x, 192, (3, 3), strides=(1, 1), padding='valid')
        x = self.ReLU(x)
    
        # Pooled feature maps will be reduced by 75%
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)
        return x
    
    def learner(self, x, n_classes):
        """ Construct the Learner
            x        : input to the learner
            n_classes: number of output classes
        """
        aux = [] # Auxiliary Outputs

        # Group 3
        x, o = self.group(x, [((64,),  (96,128),   (16, 32), (32,)),  # 3a
                              ((128,), (128, 192), (32, 96), (64,))]) # 3b
        aux += o

        # Group 4
        x, o = self.group(x, [((192,),  (96, 208), (16, 48), (64,)), # 4a
                              None, 				 # auxiliary classifier
                              ((160,), (112, 224), (24, 64), (64,)), # 4b
                              ((128,), (128, 256), (24, 64), (64,)), # 4c
                              ((112,), (144, 288), (32, 64), (64,)), # 4d
                              None,                                  # auxiliary classifier
                              ((256,), (160, 320), (32, 128), (128,))], # 4e
                              n_classes=n_classes) 
        aux += o

        # Group 5
        x, o = self.group(x, [((256,), (160, 320), (32, 128), (128,)), # 5a
                              ((384,), (192, 384), (48, 128), (128,))],# 5b
                              pooling=False) 
        aux += o
        return x, aux

    def group(self, x, blocks, pooling=True, n_classes=1000, **metaparameters):
        """ Construct an Inception group
            x         : input into the group
            blocks    : filters for each block in the group
            pooling   : whether to end the group with max pooling
            n_classes : number of classes for auxiliary classifier
        """
        aux = [] # Auxiliary Outputs

        # Construct the inception blocks (modules)
        for block in blocks:
            # Add auxiliary classifier
            if block is None:
               aux.append(self.auxiliary(x, n_classes, **metaparameters))
            else:
                x = self.inception_block(x, block[0], block[1], block[2], block[3], **metaparameters)           

        if pooling:
            x = ZeroPadding2D(padding=(1, 1))(x)
            x = MaxPooling2D((3, 3), strides=2)(x)
        return x, aux

    def inception_block(self, x, f1x1, f3x3, f5x5, fpool, **metaparameters):
        """ Construct an Inception block (module)
            x    : input to the block
            f1x1 : filters for 1x1 branch
            f3x3 : filters for 3x3 branch
            f5x5 : filters for 5x5 branch
            fpool: filters for pooling branch
        """
        # 1x1 branch
        b1x1 = self.Conv2D(x, f1x1[0], (1, 1), strides=1, padding='same', **metaparameters)
        b1x1 = self.ReLU(b1x1)

        # 3x3 branch
        # 3x3 reduction
        b3x3 = self.Conv2D(x, f3x3[0], (1, 1), strides=1, padding='same', **metaparameters)
        b3x3 = self.ReLU(b3x3)
        b3x3 = ZeroPadding2D((1,1))(b3x3)
        b3x3 = self.Conv2D(b3x3, f3x3[1], (3, 3), strides=1, padding='valid', **metaparameters)
        b3x3 = self.ReLU(b3x3)

        # 5x5 branch
        # 5x5 reduction
        b5x5 = self.Conv2D(x, f5x5[0], (1, 1), strides=1, padding='same', **metaparameters)
        b5x5 = self.ReLU(b5x5)
        b5x5 = ZeroPadding2D((1,1))(b5x5)
        b5x5 = self.Conv2D(b5x5, f5x5[1], (3, 3), strides=1, padding='valid', **metaparameters)
        b5x5 = self.ReLU(b5x5)

        # Pooling branch
        bpool = MaxPooling2D((3, 3), strides=1, padding='same')(x)
        # 1x1 projection
        bpool = self.Conv2D(bpool, fpool[0], (1, 1), strides=1, padding='same', **metaparameters)
        bpool = self.ReLU(bpool)

        # Concatenate the outputs (filters) of the branches
        x = Concatenate()([b1x1, b3x3, b5x5, bpool])
        return x

    def auxiliary(self, x, n_classes, **metaparameters):
        """ Construct the auxiliary classier
            x        : input to the auxiliary classifier
            n_classes: number of output classes
        """
        x = AveragePooling2D((5, 5), strides=(3, 3))(x)
        x = self.Conv2D(x, 128, (1, 1), strides=(1, 1), padding='same', **metaparameters)
        x = self.ReLU(x)
        x = Flatten()(x)
        x = self.Dense(x, 1024, activation=self.ReLU, **metaparameters)
        x = self.ReLU(x)
        x = Dropout(0.7)(x)
        output = self.Dense(x, n_classes, activation='softmax', **metaparameters)
        return output

    def classifier(self, x, n_classes, dropout=0.4):
        """ Construct the Classifier Group 
            x         : input to the classifier
            n_classes : number of output classes
            dropout   : percentage for dropout rate
        """
        # Save the encoding layer
        self.encoding = x
        
        # Pool at the end of all the convolutional residual blocks
        x = AveragePooling2D((7, 7))(x)
        x = Flatten()(x)

        # Save the embedding layer
        self.embedding = x

        outputs = super().classifier(x, n_classes, pooling=None, dropout=dropout)
        return outputs

# Example
# inception = InceptionV1()

def example():
    ''' Example for constructing/training a Inception V1 model on CIFAR-10
    '''
    # Example of constructing an Inception

    inception = InceptionV1(input_shape=(32, 32, 3), n_classes=10)
    inception.model.summary()
    inception.cifar10()

# Can't train on V1, since 32x32 is too small
# example()
