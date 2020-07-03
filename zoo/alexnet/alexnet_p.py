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

# AlexNet (2012) - Parallel path version for 2 GPUs
# Paper: https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, ReLU, MaxPooling2D, Dense, Flatten, Concatenate

def stem(inputs):
    """ Construct the Stem Convolutional Group 
        inputs : the input vector
    """
    # First Convolutional layer which uses an extremely large (coarse) filter
    x = Conv2D(96, (11, 11), strides=(4, 4), padding='same')(inputs)
    x = ReLU()(x)
    
    # Second Convolutional layer (1st path)
    x1 = Conv2D(128, (5, 5), strides=(1, 1), padding='same')(x)
    x1 = ReLU()(x1)
    
    # Pooled feature maps will be reduced by 75% (1st path)
    x1 = MaxPooling2D((3, 3), strides=(2, 2))(x1)
    
    # Second Convolutional layer (2nd path)
    x2 = Conv2D(128, (5, 5), strides=(1, 1), padding='same')(x)
    x2 = ReLU()(x2)
    
    # Pooled feature maps will be reduced by 75% (2nd path)
    x2 = MaxPooling2D((3, 3), strides=(2, 2))(x2)
    
    return x1, x2
    
def learner(x1, x2):
    """ Construct the Learner
        x1, x2     : inputs to the learner
    """
    # Third Convolutional layer (1st path)
    x1 = Conv2D(192, (3, 3), strides=(1, 1), padding='same')(x1)
    x1 = ReLU()(x1)
    # Pooled feature maps will be reduced by 75% (1st path)
    x1 = MaxPooling2D((3, 3), strides=(2, 2))(x1)

    # Third Convolutional layer (2nd path)
    x2 = Conv2D(192, (3, 3), strides=(1, 1), padding='same')(x2)
    x2 = ReLU()(x2)
    # Pooled feature maps will be reduced by 75% (2nd path)
    x2 = MaxPooling2D((3, 3), strides=(2, 2))(x2)
    
    # Fourth Convolutional layer (1st path)
    x1 = Conv2D(192, (3, 3), strides=(1, 1), padding='same')(x1)
    x1 = ReLU()(x1)

    # Fourth Convolutional layer (2nd path)
    x2 = Conv2D(192, (3, 3), strides=(1, 1), padding='same')(x2)
    x2 = ReLU()(x2)
    
    # Ffth Convolutional layer (1st path)
    x1 = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x1)
    x1 = ReLU()(x1)
    # Pooled feature maps will be reduced by 75% (1st path)
    x1 = MaxPooling2D((3, 3), strides=(2, 2))(x1)

    # Ffth Convolutional layer (2nd path)
    x2 = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x2)
    x2 = ReLU()(x2)
    # Pooled feature maps will be reduced by 75% (2nd path)
    x2 = MaxPooling2D((3, 3), strides=(2, 2))(x2)
    
    return x1, x2

def classifier(x1, x2, n_classes):
  """ Construct the Classifier Group 
      x1, x2    : inputs to the classifier
      n_classes : number of output classes
  """
  # Flatten into 1D vector (1st path)
  x1 = Flatten()(x1)
  # Flatten into 1D vector (2n path)
  x2 = Flatten()(x2)

  # Two dense layers of 2048 (1st path)
  x1 = Dense(2048, activation='relu')(x1)
  x1 = Dense(2048, activation='relu')(x1)
  # Two dense layers of 2048 (2nd path)
  x2 = Dense(2048, activation='relu')(x2)
  x2 = Dense(2048, activation='relu')(x2)

  # Concatenate the feature maps from the two parallel paths
  x = Concatenate()([x1, x2])

  # Final Dense Outputting Layer for the outputs
  outputs = Dense(n_classes, activation='softmax')(x)
  return outputs


# The input tensor
inputs = Input(shape=(224, 224, 3))

# The stem convolutional group
x1, x2 = stem(inputs)

# The learner
x1, x2 = learner(x1, x2)

# The classifier for 1000 classes
outputs = classifier(x1, x2, 1000)

# Instantiate the Model
# model = Model(inputs, outputs)
