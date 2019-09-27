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

# Inception v1: v1.0
# Paper: https://arxiv.org/pdf/1512.03385.pdf

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, ReLU, ZeroPadding2D
from tensorflow.keras.layers import MaxPooling2D, Dense, Add, GlobalAveragePooling2D

def stem(inputs):
    """ Construct the Stem Convolutional Group 
        inputs : the input vector
    """
    # The 224x224 images are zero padded (black - no signal) to be 230x230 images prior to the first convolution
    x = ZeroPadding2D(padding=(3, 3))(inputs)
    
    # First Convolutional layer which uses a large (coarse) filter 
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='valid', use_bias=False, kernel_initializer='he_normal')(x)
    x = ReLU()(x)
    
    # Pooled feature maps will be reduced by 75%
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    return x
    
def learner(x, n_classes):
    """ Construct the Learner
        x        : input to the learner
        n_classes: number of output classes
    """
    # Dimensiionality Expansion Groups
    x = group(x, 3, 64)
    x = group(x, 1, 128)
    # Auxiliary Classifier
    x = auxiliary(x, n_classes) 
    x = group(x, 2, 192)

    # Dimensionality Reduction Groups
    x = group(x, 1, 160)
    # Auxiliary Classifier
    x = auxiliary(x, n_classes)
    x = group(x, 2, 128)
    return x

def group(x)
    """ Construct a Residual Group 
        x         : input into the group
    """
    return x

def classifier(x, n_classes, dropout=0.4):
  """ Construct the Classifier Group 
      x         : input to the classifier
      n_classes : number of output classes
      dropout   : percentage for dropout rate
  """
  # Pool at the end of all the convolutional residual blocks
  x = AveragePooling2D((7, 7))(x)
  x = Flatten()(x)
  x = Dropout(dropout)

  # Final Dense Outputting Layer for the outputs
  outputs = Dense(n_classes, activation='softmax', kernel_initializer='glorot_uniform')(x)
  return outputs

# Meta-parameter: dropout percentage
dropout = 0.4

# The input tensor
inputs = Input(shape=(224, 224, 3))

# The stem convolutional group
x = stem(inputs)

# The learner
x = learner(x)

# The classifier for 1000 classes
outputs = classifier(x, 1000, dropout)

# Instantiate the Model
model = Model(inputs, outputs)
