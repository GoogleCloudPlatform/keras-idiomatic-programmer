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
    init_weights='glorot_uniform'

    def __init__(self, dropout=0.4, input_shape=(229, 229, 3), n_classes=1000,
                 init_weights='glorot_uniform', reg=None, relu=None):
        """ Construct an Inception V3 convolutional neural network
            dropout     : percentage of dropout rate
            input_shape : the input to the model
            n_classes   : number of output classes
            init_weights: kernel initiaklizer
            reg         : kernel regularizer
            relu        : max value for ReLU
        """
        # Configure base (super) class
        super().__init__(init_weights=init_weights, reg=reg, relu=relu)

        # The input tensor (299x299 in V3 vs 224x224 in V1/V2)
        inputs = Input(shape=input_shape)

        # The stem convolutional group
        x = self.stem(inputs)

        # The learner
        x, aux = self.learner(x, 1000)

        # The classifier
        outputs = self.classifier(x, n_classes, dropout)

        # Instantiate the Model
        self._model = Model(inputs, [outputs] + aux)

    def stem(self, inputs):
        """ Construct the Stem Convolutional Group 
            inputs : the input vector
        """
        # Coarse filter of V1 (7x7) factorized into 3 3x3.
        # First 3x3 convolution is strided
        x = Conv2D(32, (3, 3), strides=(2, 2), padding='valid', use_bias=False, kernel_initializer=self.init_weights)(inputs)
        x = BatchNormalization()(x)
        x = Composable.ReLU(x)
        x = Conv2D(32, (3, 3), strides=(1, 1), padding='valid', use_bias=False, kernel_initializer=self.init_weights)(x)
        x = BatchNormalization()(x)
        x = Composable.ReLU(x)
        # Third 3x3, filters are doubled and padding added
        x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', use_bias=False, kernel_initializer=self.init_weights)(x)
        x = BatchNormalization()(x)
        x = Composable.ReLU(x)
    
        # Pooled feature maps will be reduced by 75%
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        # 3x3 reduction
        x = Conv2D(80, (1, 1), strides=(1, 1), padding='valid', use_bias=False, kernel_initializer=self.init_weights)(x)
        x = BatchNormalization()(x)
        x = Composable.ReLU(x)
        # Dimensionality expansion
        x = Conv2D(192, (3, 3), strides=(1, 1), padding='valid', use_bias=False, kernel_initializer=self.init_weights)(x)
        x = BatchNormalization()(x)
        x = Composable.ReLU(x)

        # Pooled feature maps will be reduced by 75%
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)
        return x

    @staticmethod
    def group(x, blocks, inception=None, reduction=None, n_classes=1000, init_weights=None):
        """ Construct an Inception group
            x         : input into the group
            blocks    : filters for each block in the group
            inception : type of inception block
            reduction : whether to end the group with grid reduction
            n_classes : number of classes for auxiliary classifier
        """
        if init_weights is None:
            init_weights = InceptionV3.init_weights
            
        aux = [] # Auxiliary Outputs

        # Construct the inception blocks (modules)
        for block in blocks:
            x = inception(x, block[0], block[1], block[2], block[3], init_weights=init_weights)           

        # Add auxiliary classifier
        if n_classes:
            aux.append(InceptionV3.auxiliary(x, n_classes))
    
        # Add grid reduction
        if reduction:
            x = reduction(x)

        return x, aux

    @staticmethod
    def inception_block_A(x, f1x1, f3x3, f5x5, fpool, init_weights=None):
        """ Construct an Inception block 35x35 (module)
            x    : input to the block
            f1x1 : filters for 1x1 branch
            f3x3 : filters for 3x3 branch
            f5x5 : filters for 5x5 branch
            fpool: filters for pooling branch
        """
        if init_weights is None:
            init_weights = InceptionV1.init_weights
            
        # 1x1 branch
        b1x1 = Conv2D(f1x1[0], (1, 1), strides=1, padding='same', use_bias=False, kernel_initializer=init_weights)(x)
        b1x1 = BatchNormalization()(b1x1)
        b1x1 = Composable.ReLU(b1x1)

        # double 3x3 branch
        # 3x3 reduction
        b3x3 = Conv2D(f3x3[0], (1, 1), strides=1, padding='same', use_bias=False, kernel_initializer=init_weights)(x)
        b3x3 = BatchNormalization()(b3x3)
        b3x3 = Composable.ReLU(b3x3)
        b3x3 = Conv2D(f3x3[1], (3, 3), strides=1, padding='same', use_bias=False, kernel_initializer=init_weights)(b3x3)
        b3x3 = BatchNormalization()(b3x3)
        b3x3 = Composable.ReLU(b3x3)
        b3x3 = Conv2D(f3x3[1], (3, 3), strides=1, padding='same', use_bias=False, kernel_initializer=init_weights)(b3x3)
        b3x3 = BatchNormalization()(b3x3)
        b3x3 = Composable.ReLU(b3x3)

        # 5x5 branch
        # 5x5 reduction
        b5x5 = Conv2D(f5x5[0], (1, 1), strides=1, padding='same', use_bias=False, kernel_initializer=init_weights)(x)
        b5x5 = BatchNormalization()(b5x5)
        b5x5 = Composable.ReLU(b5x5)
        b5x5 = Conv2D(f5x5[1], (3, 3), strides=1, padding='same', use_bias=False, kernel_initializer=init_weights)(b5x5)
        b5x5 = BatchNormalization()(b5x5)
        b5x5 = Composable.ReLU(b5x5)

        # Pooling branch
        bpool = AveragePooling2D((3, 3), strides=1, padding='same')(x)
        # 1x1 projection
        bpool = Conv2D(fpool[0], (1, 1), strides=1, padding='same', use_bias=False, kernel_initializer=init_weights)(bpool)
        bpool = BatchNormalization()(bpool)
        bpool = Composable.ReLU(bpool)

        # Concatenate the outputs (filters) of the branches
        x = Concatenate()([b1x1, b3x3, b5x5, bpool])
        return x

    @staticmethod
    def inception_block_B(x, f1x1, f7x7, f7x7dbl, fpool, init_weights=None):
        """ Construct an Inception block 17x17 (module)
            x      : input to the block
            f1x1   : filters for 1x1 branch
            f7x7   : filters for 7x7 factorized asn 1x7, 7x1 branch
            f7x7dbl: filters for double 7x7 factorized as 1x7, 7x1 branch
            fpool  : filters for pooling branch
        """
        if init_weights is None:
            init_weights = InceptionV3.init_weights
            
        # 1x1 branch
        b1x1 = Conv2D(f1x1[0], (1, 1), strides=1, padding='same', use_bias=False, kernel_initializer=init_weights)(x)
        b1x1 = BatchNormalization()(b1x1)
        b1x1 = Composable.ReLU(b1x1)
    
        # 7x7 branch
        # 7x7 reduction
        b7x7 = Conv2D(f7x7[0], (1, 1), strides=1, padding='same', use_bias=False, kernel_initializer=init_weights)(x)
        b7x7 = BatchNormalization()(b7x7)
        b7x7 = Composable.ReLU(b7x7)
        # factorized 7x7
        b7x7 = Conv2D(f7x7[1], (1, 7), strides=1, padding='same', use_bias=False, kernel_initializer=init_weights)(b7x7)
        b7x7 = BatchNormalization()(b7x7)
        b7x7 = Composable.ReLU(b7x7)
        b7x7 = Conv2D(f7x7[2], (7, 1), strides=1, padding='same', use_bias=False, kernel_initializer=init_weights)(b7x7)
        b7x7 = BatchNormalization()(b7x7)
        b7x7 = Composable.ReLU(b7x7)

        # double 7x7 branch
        # 7x7 reduction
        b7x7dbl = Conv2D(f7x7dbl[0], (1, 1), strides=1, padding='same', use_bias=False, kernel_initializer=init_weights)(x)
        b7x7dbl = BatchNormalization()(b7x7dbl)
        b7x7dbl = Composable.ReLU(b7x7dbl)
        # factorized 7x7
        b7x7dbl = Conv2D(f7x7dbl[1], (1, 7), strides=1, padding='same', use_bias=False, kernel_initializer=init_weights)(b7x7dbl)
        b7x7dbl = BatchNormalization()(b7x7dbl)
        b7x7dbl = Composable.ReLU(b7x7dbl)
        b7x7dbl = Conv2D(f7x7dbl[2], (7, 1), strides=1, padding='same', use_bias=False, kernel_initializer=init_weights)(b7x7dbl)
        b7x7dbl = BatchNormalization()(b7x7dbl)
        b7x7dbl = Composable.ReLU(b7x7dbl)
        b7x7dbl = Conv2D(f7x7dbl[3], (1, 7), strides=1, padding='same', use_bias=False, kernel_initializer=init_weights)(b7x7dbl)
        b7x7dbl = BatchNormalization()(b7x7dbl)
        b7x7dbl = Composable.ReLU(b7x7dbl)
        b7x7dbl = Conv2D(f7x7dbl[4], (7, 1), strides=1, padding='same', use_bias=False, kernel_initializer=init_weights)(b7x7dbl)
        b7x7dbl = BatchNormalization()(b7x7dbl)
        b7x7dbl = Composable.ReLU(b7x7dbl)

        # Pooling branch
        bpool = AveragePooling2D((3, 3), strides=1, padding='same')(x)
        # 1x1 projection
        bpool = Conv2D(fpool[0], (1, 1), strides=1, padding='same', use_bias=False, kernel_initializer=init_weights)(bpool)
        bpool = BatchNormalization()(bpool)
        bpool = Composable.ReLU(bpool)

        # Concatenate the outputs (filters) of the branches
        x = Concatenate()([b1x1, b7x7, b7x7dbl, bpool])
        return x

    @staticmethod
    def inception_block_C(x, f1x1, f3x3, f3x3dbl, fpool, init_weights=None):
        """ Construct an Inception block 8x8 (module)
            x      : input to the block
            f1x1   : filters for 1x1 branch
            f3x3   : filters for 3x3 factorized asn 1x3, 3x1 branch
            f3x3dbl: filters for double 3x3 factorized as 1x3, 3x1 branch
            fpool  : filters for pooling branch
        """
        if init_weights is None:
            init_weights = InceptionV3.init_weights
            
        # 1x1 branch
        b1x1 = Conv2D(f1x1[0], (1, 1), strides=1, padding='same', use_bias=False, kernel_initializer=init_weights)(x)
        b1x1 = BatchNormalization()(b1x1)
        b1x1 = Composable.ReLU(b1x1)
    
        # 3x3 branch
        # 3x3 reduction
        b3x3 = Conv2D(f3x3[0], (1, 1), strides=1, padding='same', use_bias=False, kernel_initializer=init_weights)(x)
        b3x3 = BatchNormalization()(b3x3)
        b3x3 = Composable.ReLU(b3x3)
        # Split
        b3x3_1 = Conv2D(f3x3[0], (1, 3), strides=1, padding='same', use_bias=False, kernel_initializer=init_weights)(b3x3)
        b3x3_1 = BatchNormalization()(b3x3_1)
        b3x3_1 = Composable.ReLU(b3x3_1)
        b3x3_2 = Conv2D(f3x3[1], (3, 1), strides=1, padding='same', use_bias=False, kernel_initializer=init_weights)(b3x3)
        b3x3_2 = BatchNormalization()(b3x3_2)
        b3x3_2 = Composable.ReLU(b3x3_2)
        # Merge
        b3x3   = Concatenate()([b3x3_1, b3x3_2])
    
        # double 3x3 branch
        # 3x3 reduction
        b3x3dbl = Conv2D(f3x3dbl[0], (1, 1), strides=1, padding='same', use_bias=False, kernel_initializer=init_weights)(x)
        b3x3dbl = BatchNormalization()(b3x3dbl)
        b3x3dbl = Composable.ReLU(b3x3dbl)
        b3x3dbl = Conv2D(f3x3dbl[1], (3, 3), strides=1, padding='same', use_bias=False, kernel_initializer=init_weights)(b3x3dbl)
        b3x3dbl = BatchNormalization()(b3x3dbl)
        b3x3dbl = Composable.ReLU(b3x3dbl)
        # Split
        b3x3dbl_1 = Conv2D(f3x3dbl[2], (1, 3), strides=1, padding='same', use_bias=False, kernel_initializer=init_weights)(b3x3dbl)
        b3x3dbl_1 = BatchNormalization()(b3x3dbl_1)
        b3x3dbl_1 = Composable.ReLU(b3x3dbl_1)
        b3x3dbl_2 = Conv2D(f3x3dbl[3], (3, 1), strides=1, padding='same', use_bias=False, kernel_initializer=init_weights)(b3x3dbl)
        b3x3dbl_2 = BatchNormalization()(b3x3dbl_2)
        b3x3dbl_2 = Composable.ReLU(b3x3dbl_2)
        # Merge
        b3x3dbl   = Concatenate()([b3x3dbl_1, b3x3dbl_2])

        # Pooling branch
        bpool = AveragePooling2D((3, 3), strides=1, padding='same')(x)
        # 1x1 projection
        bpool = Conv2D(fpool[0], (1, 1), strides=1, padding='same', use_bias=False, kernel_initializer=init_weights)(bpool)
        bpool = BatchNormalization()(bpool)
        bpool = Composable.ReLU(bpool)

        # Concatenate the outputs (filters) of the branches
        x = Concatenate()([b1x1, b3x3, b3x3dbl, bpool])
        return x

    @staticmethod
    def grid_reduction_A(x, f3x3=384, f3x3dbl=(64, 96, 96), init_weights=None):
        """ Construct the Grid Reduction block
            x       : input to the block
            f3x3    : filter size for 3x3 branch
            f3x3dbl : filter sizes for double 3x3 branch
        """
        if init_weights is None:
            init_weights = InceptionV3.init_weights
            
        # 3x3 branch
        # grid reduction
        b3x3 = Conv2D(f3x3, (3, 3), strides=2, padding='valid', use_bias=False, kernel_initializer=init_weights)(x)
        b3x3 = BatchNormalization()(b3x3)
        b3x3 = Composable.ReLU(b3x3)

        # double 3x3 branch
        # 3x3 reduction
        b3x3dbl = Conv2D(f3x3dbl[0], (1, 1), strides=1, padding='same', use_bias=False, kernel_initializer=init_weights)(x)
        b3x3dbl = BatchNormalization()(b3x3dbl)
        b3x3dbl = Composable.ReLU(b3x3dbl)
        b3x3dbl = Conv2D(f3x3dbl[1], (3, 3), strides=1, padding='same', use_bias=False, kernel_initializer=init_weights)(b3x3dbl)
        b3x3dbl = BatchNormalization()(b3x3dbl)
        b3x3dbl = Composable.ReLU(b3x3dbl)
        # grid reduction
        b3x3dbl = Conv2D(f3x3dbl[1], (3, 3), strides=2, padding='valid', use_bias=False, kernel_initializer=init_weights)(b3x3dbl)
        b3x3dbl = BatchNormalization()(b3x3dbl)
        b3x3dbl = Composable.ReLU(b3x3dbl)

        # pool branch
        bpool   = MaxPooling2D((3, 3), strides=2)(x)

        # Concatenate the outputs (filters) of the branches
        x = Concatenate()([b3x3, b3x3dbl, bpool])
        return x

    @staticmethod
    def grid_reduction_B(x, f3x3=(192, 320), f7x7=(192, 192, 192, 192), init_weights=None):
        """ Construct the Grid Reduction block
            x       : input to the block
            f3x3    : filter size for 3x3 branch
            f7x7    : filter sizes for 7x7 + 3x3 branch
        """
        if init_weights is None:
            init_weights = InceptionV3.init_weights
            
        # 3x3 branch
        # 3x3 reduction
        b3x3 = Conv2D(f3x3[0], (1, 1), strides=1, padding='same', use_bias=False, kernel_initializer=init_weights)(x)
        b3x3 = BatchNormalization()(b3x3)
        b3x3 = Composable.ReLU(b3x3)
        # grid reduction
        b3x3 = Conv2D(f3x3[1], (3, 3), strides=2, padding='valid', use_bias=False, kernel_initializer=init_weights)(b3x3)
        b3x3 = BatchNormalization()(b3x3)
        b3x3 = Composable.ReLU(b3x3)

        # 7x7 (factorized as 1x7, 7x1) + 3x3 branch
        # 7x7 reduction
        b7x7 = Conv2D(f7x7[0], (1, 1), strides=1, padding='same', use_bias=False, kernel_initializer=init_weights)(x)
        b7x7 = BatchNormalization()(b7x7)
        b7x7 = Composable.ReLU(b7x7)
        b7x7 = Conv2D(f7x7[1], (1, 7), strides=1, padding='same', use_bias=False, kernel_initializer=init_weights)(b7x7)
        b7x7 = BatchNormalization()(b7x7)
        b7x7 = Composable.ReLU(b7x7)
        b7x7 = Conv2D(f7x7[2], (7, 1), strides=1, padding='same', use_bias=False, kernel_initializer=init_weights)(b7x7)
        b7x7 = BatchNormalization()(b7x7)
        b7x7 = Composable.ReLU(b7x7)
        # grid reduction
        b7x7 = Conv2D(f7x7[3], (3, 3), strides=2, padding='valid', use_bias=False, kernel_initializer=init_weights)(b7x7)
        b7x7 = BatchNormalization()(b7x7)
        b7x7 = Composable.ReLU(b7x7)

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
        x, o = InceptionV3.group(x, [((64,), (64, 96), (48, 64), (32,)),
                                     ((64,), (64, 96), (48, 64), (64,)),
                                     ((64,), (64, 96), (48, 64), (64,))
                                    ], inception=InceptionV3.inception_block_A, reduction=InceptionV3.grid_reduction_A) 
        aux += o

        # Group B (17x17)
        x, o = InceptionV3.group(x, [((192,), (128, 128, 192), (128, 128, 128, 128, 192), (192,)),
                                     ((192,), (160, 160, 192), (160, 160, 160, 160, 192), (192,)),
                                     ((192,), (160, 160, 192), (160, 160, 160, 160, 192), (192,)),
                                     ((192,), (192, 192, 192), (192, 192, 192, 192, 192), (192,))
                                    ], inception=InceptionV3.inception_block_B, reduction=InceptionV3.grid_reduction_B, n_classes=n_classes) 
        aux += o

        # Group C (8x8)
        x, o = InceptionV3.group(x, [((320,), (384, 384, 384), (448, 384, 384, 384), (192,)),
                                     ((320,), (384, 384, 384), (448, 384, 384, 384), (192,))
                                    ], inception=InceptionV3.inception_block_C) 
        aux += o
        return x, aux

    @staticmethod
    def auxiliary(x, n_classes, init_weights=None):
        """ Construct the auxiliary classier
            x        : input to the auxiliary classifier
            n_classes: number of output classes
        """
        if init_weights is None:
            init_weights = InceptionV3.init_weights
            
        x = AveragePooling2D((5, 5), strides=(3, 3))(x)
        x = Conv2D(128, (1, 1), strides=(1, 1), use_bias=False, kernel_initializer=init_weights)(x)
        x = BatchNormalization()(x)
        x = Composable.ReLU(x)
        # filter will be 5x5 for V3
        x = Conv2D(768, x.shape[1:3].as_list(), strides=(1, 1), use_bias=False, kernel_initializer=init_weights)(x)
        x = Flatten()(x)
        output = Dense(n_classes, activation='softmax', kernel_initializer='glorot_uniform')(x)
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

        # Final Dense Outputting Layer for the outputs
        x = Dense(n_classes, kernel_initializer=self.init_weights)(x)
        # Save the pre-activation probabilities
        self.probabilities = x
        outputs = Activation('softmax')(x)
        # Save the post-activation probabilities
        self.softmax = outputs
        return outputs

# Example
# inception = InceptionV3()
