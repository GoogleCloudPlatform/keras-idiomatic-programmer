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

# Post-Upsampling Super Resolution CNN (SRCNN) (2016)
# Paper: https://arxiv.org/pdf/1501.00092.pdf

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Conv2DTranspose, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

import sys
sys.path.append('../')
from models_c import Composable

class SRCNNPost(Composable):
    ''' Construct a Post Upsampling Super Resolution CNN '''
    # Meta-parameter: 
    groups = [ { 'n_filters': 32, 'n_filters' : 64 } ]

    # Initial Hyperparameters
    hyperparameters = { 'initializer': 'he_normal',
                        'regularizer': None,
                        'relu_clip'  : None,
                        'bn_epsilon' : None,
                        'use_bias'   : False
                      }

    def __init__(self, groups=None ,
                 input_shape=(32, 32, 3), include_top=True,
                 **hyperparameters):
        """ Construct a Wids Residual (Convolutional Neural) Network 
            groups      : metaparameter for group configuration
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

        if groups is None:
            groups = self.groups

        # The input tensor
        inputs = Input(input_shape)

        # The stem convolutional group
        x = self.stem(inputs)

        # The learner
        outputs = self.learner(x, groups)

        # The reconstruction
        if include_top:
             outputs = self.decoder(outputs)

        # Instantiate the Model
        self._model = Model(inputs, outputs)

    def stem(self, inputs):
        """ Construct the Stem Convolutional Group 
            inputs : the input tensor
        """
        # n1, dimensionality expansion with large coarse filter
        x = self.Conv2D(inputs, 16, (3, 3), padding='same')
        x = self.BatchNormalization(x)
        x = self.ReLU(x)
        return x

    def learner(self, x, groups):
        """ Construct the Learner 
            x      : input to the learner
            groups : group configuration
        """
        for group in groups:
            n_filters = group['n_filters' ]
            x = self.Conv2D(x, n_filters, (3, 3), padding='same')
            x = self.BatchNormalization(x)
            x = self.ReLU(x)
        return x

    def decoder(self, x):
        """ Construct the Decoder
            x    : input to the decoder
        """
        # reconstruction
        x = self.Conv2DTranspose(x, 3, (3, 3), strides=2, padding='same')
        x = self.BatchNormalization(x)
        x = Activation('sigmoid')(x)
        return x

# Example
# srcnn = SRCNNPost()
