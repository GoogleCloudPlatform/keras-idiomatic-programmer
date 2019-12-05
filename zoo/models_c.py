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

import tensorflow as tf
from tensorflow.keras.layers import ReLU
from tensorflow.keras.regularizers import l2

class Composable(object):
    ''' Composable base (super) class for Models '''
    init_weights = 'he_normal'	# weight initialization
    reg          = l2(0.001)    # kernel regularizer
    relu         = None         # ReLU max value

    def __init__(self, init_weights=None, reg=None, relu=None):
        """ Constructor
            init_weights :
            relu         :
        """
        if init_weights is not None:
            self.init_weights = init_weights
        if reg is not None:
            self.reg = reg
        if relu is not None:
            self.relu = relu

    @staticmethod
    def ReLU(x):
        """ Construct ReLU activation function
        x  : input to activation function
        """
        x = ReLU(Composable.relu)(x)
        return x
	
