# Copyright 2020 Google LLC
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

# Super Resolution CNN (SRCNN) (2016)
# Paper: https://arxiv.org/pdf/1501.00092.pdf

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Conv2DTranspose, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

import sys
sys.path.append('../')
from models_c import Composable

class SRCNN(Composable):
    ''' Construct a Super Resolution CNN '''
    # Meta-parameter: filter sizes for n1, n2 and n3 convolutions
    f1 = 9
    f2 = 1
    f3 = 5

    # Initial Hyperparameters
    hyperparameters = { 'initializer': 'he_normal',
                        'regularizer': None,
                        'relu_clip'  : None,
                        'bn_epsilon' : None,
                        'use_bias'   : False
                      }

    def __init__(self,  
                 input_shape=(32, 32, 3), include_top=True,
                 f1 = 9, f2=1, f3=5,
                 **hyperparameters):
        """ Construct a Wids Residual (Convolutional Neural) Network 
            f1, f2, f3  : number of filters for convolutional layers n1, n2 and n3
            input_shape : input shape
            include_top : include the reconstruction component
            initializer : kernel initialization
            regularizer : kernel regularization
            relu_clip   : max value for ReLU
            bn_epsilon  : epsilon for batch norm
            use_bias    : whether use bias in conjunction with batch norm
        """
        # Configure base (super) class
        Composable.__init__(self, input_shape, include_top, self.hyperparameters, **hyperparameters)

        # The input tensor
        inputs = Input(input_shape)

        # The stem convolutional group
        x = self.stem(inputs, f1)

        # The encoder
        outputs = self.encoder(x, f2)

        # The reconstruction
        if include_top:
             outputs = self.reconstruction(outputs, f3)

        # Instantiate the Model
        self._model = Model(inputs, outputs)

    def stem(self, inputs, f1):
        """ Construct the Stem Convolutional Group 
            inputs : the input tensor
            f1     : filter size
        """
        # n1, dimensionality expansion with large coarse filter
        x = self.Conv2D(inputs, 64, (f1, f1), padding='same')
        x = self.BatchNormalization(x)
        x = self.ReLU(x)
        return x

    def encoder(self, x, f2):
        """ Construct the Encoder
            x  : the input to the encoder
            f2 : the filter size
        """
        # n2, 1x1 bottleneck convolution
        x = self.Conv2D(x, 32, (f2, f2), padding='same')
        x = self.BatchNormalization(x)
        x = self.ReLU(x)
        return x

    def reconstruction(self, x, f3):
        """ Construct the Encoder
            x  : the input to the reconstruction
            f3 : the filter size
        """
        # n3, reconstruction convolution
        x = self.Conv2D(x, 3, (f3, f3), padding='same')
        x = self.BatchNormalization(x)
        outputs = Activation('sigmoid')(x)
        return outputs

# Example
# srcnn = SRCNN(f1=9, f2=1, f3=5)
