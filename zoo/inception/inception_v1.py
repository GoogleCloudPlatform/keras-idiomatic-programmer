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

# Inception v1 (GoogleNet)
# Paper: https://arxiv.org/pdf/1409.4842.pdf

### IN PROGRESS

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D

def stem(inputs):
    """ Construct Stem Group 
        inputs : input vector
    """
    # Zero pad the image, such that the 224x224 becomes 230x230
    x = ZeroPadding2D((3, 3))(inputs)
    # Do convolution with a coarse filter (7x7)
    x = Conv2D( 64, (7, 7), strides=2, activation='relu')(x)
    
    # Reduce the feature map size by 75%
    x = ZeroPadding2D((1, 1))(inputs)
    x = MaxPooling2D((3, 3), strides=2)(x)
    
    x = Conv2D( 192, (1, 1), strides=1, activation='relu')(x)
    x = Conv2D( 192, (3, 3), strides=1, activation='relu')(x)
    
    # Reduce the feature map size by 75%
    x = MaxPooling2D((3, 3), strides=2)(x)
    return x
    
inputs = Input((224, 224, 3))

x = stem(inputs)

model = Model(inputs, x)
model.summary()
    
