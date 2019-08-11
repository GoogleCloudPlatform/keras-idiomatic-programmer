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

# VGG 19 (2014)
# Paper: https://arxiv.org/pdf/1409.1556.pdf

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def stem(inputs):
    """ Create Stem Convolutional Group
        inputs : the input vector
    """
    x = Conv2D(64, (3, 3), strides=(1, 1), padding="same", activation="relu")(inputs)
    return x

def conv_block(x, n_layers, n_filters):
    """ Create Convolutional Block
        x        : input to the block
        n_layers : number of convolutional layers
        n_filters: number of filters
    """
    # Block of convolutional layers
    for n in range(n_layers):
        x = Conv2D(n_filters, (3, 3), strides=(1, 1), padding="same", activation="relu")(x)
        
    # Max pooling at the end of the block
    x = MaxPooling2D(2, strides=(2, 2))(x)
    return x
    
def classifier(x, n_classes):
    """ Create Classifier
        x         : input to the classifier
        n_classes : number of output classes
    """
    # Flatten the feature maps
    x = Flatten()(x)
    
    # Two fully connected dense layers
    x = Dense(4096, activation='relu')(x)
    x = Dense(4096, activation='relu')(x)

    # Output layer for classification 
    x = Dense(n_classes, activation='softmax')(x)
    return x
 
# The input vector 
inputs = Input( (224, 224, 3) )

# The stem group
x = stem(inputs)

# The convolutional blocks
x = conv_block(x, 1, 64)
x = conv_block(x, 2, 128)
x = conv_block(x, 4, 256)
x = conv_block(x, 4, 512)
x = conv_block(x, 4, 512)

# The classifier
outputs = classifier(x, 1000)

# Instantiate the Model
model = Model(inputs, outputs)
model.summary()

