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

# Inception v3:
# Trainable params: 33,666,688
# Paper: https://arxiv.org/pdf/1512.00567.pdf

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, ReLU, ZeroPadding2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import MaxPooling2D, Dense, Concatenate, AveragePooling2D, Activation

import sys
sys.path.append('../')
from models_c import Composable

class InceptionV3(Composable):
    """ Construct an Inception V3 convolutional neural network """

    # Initial Hyperparameters
    hyperparameters = { 'initializer': 'he_normal',
                        'regularizer': None,
                        'relu_clip'  : None,
                        'bn_epsilon' : None,
                        'use_bias'   : False
                      }

    def __init__(self, dropout=0.4, 
                 input_shape=(229, 229, 3), n_classes=1000, include_top=True,
                 **hyperparameters):
        """ Construct an Inception V3 convolutional neural network
            dropout     : percentage of dropout rate
            input_shape : the input to the model
            n_classes   : number of output classes
            include_top : whether to include the classifier
            initializer : kernel initiaklizer
            regularizer : kernel regularizer
            relu_clip   : max value for ReLU
            bn_epsilon  : epsilon for batch norm
            use_bias    : whether to use bias
        """
        # Configure base (super) class
        Composable.__init__(self, input_shape, include_top, self.hyperparameters, **hyperparameters)

        # The input tensor (299x299 in V3 vs 224x224 in V1/V2)
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
        # Coarse filter of V1 (7x7) factorized into 3 3x3.
        # First 3x3 convolution is strided
        x = self.Conv2D(inputs, 32, (3, 3), strides=(2, 2), padding='valid')
        x = self.BatchNormalization(x)
        x = self.ReLU(x)
        x = self.Conv2D(x, 32, (3, 3), strides=(1, 1), padding='valid')
        x = self.BatchNormalization(x)
        x = self.ReLU(x)
        # Third 3x3, filters are doubled and padding added
        x = self.Conv2D(x, 64, (3, 3), strides=(1, 1), padding='same')
        x = self.BatchNormalization(x)
        x = self.ReLU(x)
    
        # Pooled feature maps will be reduced by 75%
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        # 3x3 reduction
        x = self.Conv2D(x, 80, (1, 1), strides=(1, 1), padding='valid')
        x = self.BatchNormalization(x)
        x = self.ReLU(x)
        # Dimensionality expansion
        x = self.Conv2D(x, 192, (3, 3), strides=(1, 1), padding='valid')
        x = self.BatchNormalization(x)
        x = self.ReLU(x)

        # Pooled feature maps will be reduced by 75%
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)
        return x

    def group(self, x, blocks, inception=None, reduction=None, n_classes=1000, **metaparameters):
        """ Construct an Inception group
            x         : input into the group
            blocks    : filters for each block in the group
            inception : type of inception block
            reduction : whether to end the group with grid reduction
            n_classes : number of classes for auxiliary classifier
        """           
        aux = [] # Auxiliary Outputs

        # Construct the inception blocks (modules)
        for block in blocks:
            x = inception(x, block[0], block[1], block[2], block[3], **metaparameters)           

        # Add auxiliary classifier
        if n_classes:
            aux.append(self.auxiliary(x, n_classes, **metaparameters))
    
        # Add grid reduction
        if reduction:
            x = reduction(x)

        return x, aux

    def inception_block_A(self, x, f1x1, f3x3, f5x5, fpool, **metaparameters):
        """ Construct an Inception block 35x35 (module)
            x    : input to the block
            f1x1 : filters for 1x1 branch
            f3x3 : filters for 3x3 branch
            f5x5 : filters for 5x5 branch
            fpool: filters for pooling branch
        """           
        # 1x1 branch
        b1x1 = self.Conv2D(x, f1x1[0], (1, 1), strides=1, padding='same', **metaparameters)
        b1x1 = self.BatchNormalization(b1x1)
        b1x1 = self.ReLU(b1x1)

        # double 3x3 branch
        # 3x3 reduction
        b3x3 = self.Conv2D(x, f3x3[0], (1, 1), strides=1, padding='same', **metaparameters)
        b3x3 = self.BatchNormalization(b3x3)
        b3x3 = self.ReLU(b3x3)
        b3x3 = self.Conv2D(b3x3, f3x3[1], (3, 3), strides=1, padding='same', **metaparameters)
        b3x3 = self.BatchNormalization(b3x3)
        b3x3 = self.ReLU(b3x3)
        b3x3 = self.Conv2D(b3x3, f3x3[1], (3, 3), strides=1, padding='same', **metaparameters)
        b3x3 = self.BatchNormalization(b3x3)
        b3x3 = self.ReLU(b3x3)

        # 5x5 branch
        # 5x5 reduction
        b5x5 = self.Conv2D(x, f5x5[0], (1, 1), strides=1, padding='same', **metaparameters)
        b5x5 = self.BatchNormalization(b5x5)
        b5x5 = self.ReLU(b5x5)
        b5x5 = self.Conv2D(b5x5, f5x5[1], (3, 3), strides=1, padding='same', **metaparameters)
        b5x5 = self.BatchNormalization(b5x5)
        b5x5 = self.ReLU(b5x5)

        # Pooling branch
        bpool = AveragePooling2D((3, 3), strides=1, padding='same')(x)
        # 1x1 projection
        bpool = self.Conv2D(bpool, fpool[0], (1, 1), strides=1, padding='same', **metaparameters)
        bpool = self.BatchNormalization(bpool)
        bpool = self.ReLU(bpool)

        # Concatenate the outputs (filters) of the branches
        x = Concatenate()([b1x1, b3x3, b5x5, bpool])
        return x

    def inception_block_B(self, x, f1x1, f7x7, f7x7dbl, fpool, **metaparameters):
        """ Construct an Inception block 17x17 (module)
            x      : input to the block
            f1x1   : filters for 1x1 branch
            f7x7   : filters for 7x7 factorized asn 1x7, 7x1 branch
            f7x7dbl: filters for double 7x7 factorized as 1x7, 7x1 branch
            fpool  : filters for pooling branch
        """     
        # 1x1 branch
        b1x1 = self.Conv2D(x, f1x1[0], (1, 1), strides=1, padding='same', **metaparameters)
        b1x1 = self.BatchNormalization(b1x1)
        b1x1 = self.ReLU(b1x1)
    
        # 7x7 branch
        # 7x7 reduction
        b7x7 = self.Conv2D(x, f7x7[0], (1, 1), strides=1, padding='same', **metaparameters)
        b7x7 = self.BatchNormalization(b7x7)
        b7x7 = self.ReLU(b7x7)
        # factorized 7x7
        b7x7 = self.Conv2D(b7x7, f7x7[1], (1, 7), strides=1, padding='same', **metaparameters)
        b7x7 = self.BatchNormalization(b7x7)
        b7x7 = self.ReLU(b7x7)
        b7x7 = self.Conv2D(b7x7, f7x7[2], (7, 1), strides=1, padding='same', **metaparameters)
        b7x7 = self.BatchNormalization(b7x7)
        b7x7 = self.ReLU(b7x7)

        # double 7x7 branch
        # 7x7 reduction
        b7x7dbl = self.Conv2D(x, f7x7dbl[0], (1, 1), strides=1, padding='same', **metaparameters)
        b7x7dbl = self.BatchNormalization(b7x7dbl)
        b7x7dbl = self.ReLU(b7x7dbl)
        # factorized 7x7
        b7x7dbl = self.Conv2D(b7x7dbl, f7x7dbl[1], (1, 7), strides=1, padding='same', **metaparameters)
        b7x7dbl = self.BatchNormalization(b7x7dbl)
        b7x7dbl = self.ReLU(b7x7dbl)
        b7x7dbl = self.Conv2D(b7x7dbl, f7x7dbl[2], (7, 1), strides=1, padding='same', **metaparameters)
        b7x7dbl = self.BatchNormalization(b7x7dbl)
        b7x7dbl = self.ReLU(b7x7dbl)
        b7x7dbl = self.Conv2D(b7x7dbl, f7x7dbl[3], (1, 7), strides=1, padding='same', **metaparameters)
        b7x7dbl = self.BatchNormalization(b7x7dbl)
        b7x7dbl = self.ReLU(b7x7dbl)
        b7x7dbl = self.Conv2D(b7x7dbl, f7x7dbl[4], (7, 1), strides=1, padding='same', **metaparameters)
        b7x7dbl = self.BatchNormalization(b7x7dbl)
        b7x7dbl = self.ReLU(b7x7dbl)

        # Pooling branch
        bpool = AveragePooling2D((3, 3), strides=1, padding='same')(x)
        # 1x1 projection
        bpool = self.Conv2D(bpool, fpool[0], (1, 1), strides=1, padding='same', **metaparameters)
        bpool = self.BatchNormalization(bpool)
        bpool = self.ReLU(bpool)

        # Concatenate the outputs (filters) of the branches
        x = Concatenate()([b1x1, b7x7, b7x7dbl, bpool])
        return x

    def inception_block_C(self, x, f1x1, f3x3, f3x3dbl, fpool, **metaparameters):
        """ Construct an Inception block 8x8 (module)
            x      : input to the block
            f1x1   : filters for 1x1 branch
            f3x3   : filters for 3x3 factorized asn 1x3, 3x1 branch
            f3x3dbl: filters for double 3x3 factorized as 1x3, 3x1 branch
            fpool  : filters for pooling branch
        """
        # 1x1 branch
        b1x1 = self.Conv2D(x, f1x1[0], (1, 1), strides=1, padding='same', **metaparameters)
        b1x1 = self.BatchNormalization(b1x1)
        b1x1 = self.ReLU(b1x1)
    
        # 3x3 branch
        # 3x3 reduction
        b3x3 = self.Conv2D(x, f3x3[0], (1, 1), strides=1, padding='same', **metaparameters)
        b3x3 = self.BatchNormalization(b3x3)
        b3x3 = self.ReLU(b3x3)
        # Split
        b3x3_1 = self.Conv2D(b3x3, f3x3[0], (1, 3), strides=1, padding='same', **metaparameters)
        b3x3_1 = self.BatchNormalization(b3x3_1)
        b3x3_1 = self.ReLU(b3x3_1)
        b3x3_2 = self.Conv2D(b3x3, f3x3[1], (3, 1), strides=1, padding='same', **metaparameters)
        b3x3_2 = self.BatchNormalization(b3x3_2)
        b3x3_2 = self.ReLU(b3x3_2)
        # Merge
        b3x3   = Concatenate()([b3x3_1, b3x3_2])
    
        # double 3x3 branch
        # 3x3 reduction
        b3x3dbl = self.Conv2D(x, f3x3dbl[0], (1, 1), strides=1, padding='same', **metaparameters)
        b3x3dbl = self.BatchNormalization(b3x3dbl)
        b3x3dbl = self.ReLU(b3x3dbl)
        b3x3dbl = self.Conv2D(b3x3dbl, f3x3dbl[1], (3, 3), strides=1, padding='same', **metaparameters)
        b3x3dbl = self.BatchNormalization(b3x3dbl)
        b3x3dbl = self.ReLU(b3x3dbl)
        # Split
        b3x3dbl_1 = self.Conv2D(b3x3dbl, f3x3dbl[2], (1, 3), strides=1, padding='same', **metaparameters)
        b3x3dbl_1 = self.BatchNormalization(b3x3dbl_1)
        b3x3dbl_1 = self.ReLU(b3x3dbl_1)
        b3x3dbl_2 = self.Conv2D(b3x3dbl, f3x3dbl[3], (3, 1), strides=1, padding='same', **metaparameters)
        b3x3dbl_2 = self.BatchNormalization(b3x3dbl_2)
        b3x3dbl_2 = self.ReLU(b3x3dbl_2)
        # Merge
        b3x3dbl   = Concatenate()([b3x3dbl_1, b3x3dbl_2])

        # Pooling branch
        bpool = AveragePooling2D((3, 3), strides=1, padding='same')(x)
        # 1x1 projection
        bpool = self.Conv2D(bpool, fpool[0], (1, 1), strides=1, padding='same', **metaparameters)
        bpool = self.BatchNormalization(bpool)
        bpool = self.ReLU(bpool)

        # Concatenate the outputs (filters) of the branches
        x = Concatenate()([b1x1, b3x3, b3x3dbl, bpool])
        return x

    def grid_reduction_A(self, x, f3x3=384, f3x3dbl=(64, 96, 96), **metaparameters):
        """ Construct the Grid Reduction block
            x       : input to the block
            f3x3    : filter size for 3x3 branch
            f3x3dbl : filter sizes for double 3x3 branch
        """         
        # 3x3 branch
        # grid reduction
        b3x3 = self.Conv2D(x, f3x3, (3, 3), strides=2, padding='valid', **metaparameters)
        b3x3 = self.BatchNormalization(b3x3)
        b3x3 = self.ReLU(b3x3)

        # double 3x3 branch
        # 3x3 reduction
        b3x3dbl = self.Conv2D(x, f3x3dbl[0], (1, 1), strides=1, padding='same', **metaparameters)
        b3x3dbl = self.BatchNormalization(b3x3dbl)
        b3x3dbl = self.ReLU(b3x3dbl)
        b3x3dbl = self.Conv2D(b3x3dbl, f3x3dbl[1], (3, 3), strides=1, padding='same', **metaparameters)
        b3x3dbl = self.BatchNormalization(b3x3dbl)
        b3x3dbl = self.ReLU(b3x3dbl)
        # grid reduction
        b3x3dbl = self.Conv2D(b3x3dbl, f3x3dbl[1], (3, 3), strides=2, padding='valid', **metaparameters)
        b3x3dbl = self.BatchNormalization(b3x3dbl)
        b3x3dbl = self.ReLU(b3x3dbl)

        # pool branch
        bpool   = MaxPooling2D((3, 3), strides=2)(x)

        # Concatenate the outputs (filters) of the branches
        x = Concatenate()([b3x3, b3x3dbl, bpool])
        return x

    def grid_reduction_B(self, x, f3x3=(192, 320), f7x7=(192, 192, 192, 192), **metaparameters):
        """ Construct the Grid Reduction block
            x       : input to the block
            f3x3    : filter size for 3x3 branch
            f7x7    : filter sizes for 7x7 + 3x3 branch
        """         
        # 3x3 branch
        # 3x3 reduction
        b3x3 = self.Conv2D(x, f3x3[0], (1, 1), strides=1, padding='same', **metaparameters)
        b3x3 = self.BatchNormalization(b3x3)
        b3x3 = self.ReLU(b3x3)
        # grid reduction
        b3x3 = self.Conv2D(b3x3, f3x3[1], (3, 3), strides=2, padding='valid', **metaparameters)
        b3x3 = self.BatchNormalization(b3x3)
        b3x3 = self.ReLU(b3x3)

        # 7x7 (factorized as 1x7, 7x1) + 3x3 branch
        # 7x7 reduction
        b7x7 = self.Conv2D(x, f7x7[0], (1, 1), strides=1, padding='same', **metaparameters)
        b7x7 = self.BatchNormalization(b7x7)
        b7x7 = self.ReLU(b7x7)
        b7x7 = self.Conv2D(b7x7, f7x7[1], (1, 7), strides=1, padding='same', **metaparameters)
        b7x7 = self.BatchNormalization(b7x7)
        b7x7 = self.ReLU(b7x7)
        b7x7 = self.Conv2D(b7x7, f7x7[2], (7, 1), strides=1, padding='same', **metaparameters)
        b7x7 = self.BatchNormalization(b7x7)
        b7x7 = self.ReLU(b7x7)
        # grid reduction
        b7x7 = self.Conv2D(b7x7, f7x7[3], (3, 3), strides=2, padding='valid', **metaparameters)
        b7x7 = self.BatchNormalization(b7x7)
        b7x7 = self.ReLU(b7x7)

        # pool branch
        bpool   = MaxPooling2D((3, 3), strides=2)(x)

        # Concatenate the outputs (filters) of the branches
        x = Concatenate()([b3x3, b7x7, bpool])
        return x
    
    def learner(self, x, n_classes):
        """ Construct the Learner
            x        : input to the learner
            n_classes: number of output classes
        """
        aux = [] # Auxiliary Outputs

        # Group A (35x35)
        x, o = self.group(x, [((64,), (64, 96), (48, 64), (32,)),
                              ((64,), (64, 96), (48, 64), (64,)),
                              ((64,), (64, 96), (48, 64), (64,))
                             ], inception=self.inception_block_A, reduction=self.grid_reduction_A) 
        aux += o

        # Group B (17x17)
        x, o = self.group(x, [((192,), (128, 128, 192), (128, 128, 128, 128, 192), (192,)),
                              ((192,), (160, 160, 192), (160, 160, 160, 160, 192), (192,)),
                              ((192,), (160, 160, 192), (160, 160, 160, 160, 192), (192,)),
                              ((192,), (192, 192, 192), (192, 192, 192, 192, 192), (192,))
                             ], inception=self.inception_block_B, reduction=self.grid_reduction_B, n_classes=n_classes) 
        aux += o

        # Group C (8x8)
        x, o = self.group(x, [((320,), (384, 384, 384), (448, 384, 384, 384), (192,)),
                              ((320,), (384, 384, 384), (448, 384, 384, 384), (192,))
                             ], inception=self.inception_block_C) 
        aux += o
        return x, aux

    def auxiliary(self, x, n_classes, **metaparameters):
        """ Construct the auxiliary classier
            x        : input to the auxiliary classifier
            n_classes: number of output classes
        """           
        x = AveragePooling2D((5, 5), strides=(3, 3))(x)
        x = self.Conv2D(x, 128, (1, 1), strides=(1, 1), **metaparameters)
        x = self.BatchNormalization(x)
        x = self.ReLU(x)
        # filter will be 5x5 for V3
        x = self.Conv2D(x, 768, x.shape[1:3].as_list(), strides=(1, 1), **metaparameters)
        x = Flatten()(x)
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
        # Will be 8x8 in V3
        x = AveragePooling2D(x.shape[1:3].as_list())(x)
        x = Dropout(dropout)(x)
        x = Flatten()(x)

        # Save the embedding layer
        self.embedding = x

        outputs = super().classifier(x, n_classes, pooling=None)
        return outputs

# Example
# inception = InceptionV3()

def example():
    ''' Example for constructing/training a Inception V3 model on CIFAR-10
    '''
    # Example of constructing an Inception

    inception = InceptionV3(input_shape=(32, 32, 3), n_classes=10)
    inception.model.summary()
    inception.cifar10()

# Can't train on V3, since 32x32 is too small
# example()
