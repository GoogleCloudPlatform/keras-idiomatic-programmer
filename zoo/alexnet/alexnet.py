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

# AlexNet (2012) - simplified as a single path
# Paper: https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, ReLU, MaxPooling2D, Dense, Flatten

def stem(inputs):
    """ Construct the Stem Convolutional Group 
        inputs : the input vector
    """
    # First Convolutional layer which uses an extremely large (coarse) filter
    x = Conv2D(96, (11, 11), strides=(4, 4), padding='same')(inputs)
    x = ReLU()(x)
    
    # Second Convolutional layer
    x = Conv2D(256, (5, 5), strides=(1, 1), padding='same')(x)
    x = ReLU()(x)
    
    # Pooled feature maps will be reduced by 75%
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    return x
    
def learner(x):
    """ Construct the Learner
        x     : input to the learner
    """
    # Third Convolutional layer
    x = Conv2D(384, (3, 3), strides=(1, 1), padding='same')(x)
    x = ReLU()(x)
    
    # Pooled feature maps will be reduced by 75%
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    
    # Fourth Convolutional layer
    x = Conv2D(384, (3, 3), strides=(1, 1), padding='same')(x)
    x = ReLU()(x)
    
    # Ffth Convolutional layer
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
    x = ReLU()(x)
    
    # Pooled feature maps will be reduced by 75%
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    
    return x

def classifier(x, n_classes):
  """ Construct the Classifier Group 
      x         : input to the classifier
      n_classes : number of output classes
  """
  # Flatten into 1D vector
  x = Flatten()(x)

  # Two dense layers of 4096
  x = Dense(4096, activation='relu')(x)
  x = Dense(4096, activation='relu')(x)

  # Final Dense Outputting Layer for the outputs
  outputs = Dense(n_classes, activation='softmax')(x)
  return outputs


# The input tensor
inputs = Input(shape=(224, 224, 3))

# The stem convolutional group
x = stem(inputs)

# The learner
x = learner(x)

# The classifier for 1000 classes
outputs = classifier(x, 1000)

# Instantiate the Model
# model = Model(inputs, outputs)
