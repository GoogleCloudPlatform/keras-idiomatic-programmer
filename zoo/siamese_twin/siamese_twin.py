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

# Siamese Net for One-Shot Image Classification (Koch, et. al.)
# Paper: https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pd

import tensorflow as tf
from tensorflow.keras import Input, Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Lambda
from tensorflow.keras.initializers import RandomNormal
import tensorflow.keras.backend as K

def twin(input_shape):
    ''' Construct the model for both twins of the Siamese (connected) Network
        input_shape : input shape for input vector
    '''
    global dense_weights, biases
    
    model = Sequential()
    
    # The weights for the convolutional layers are initialized from a normal distribution
    # with a zero_mean and standard deviation of 10e-2
    conv_weights = RandomNormal(mean=0.0, stddev=10e-2)
    
    # The weights for the dense layers are initialized from a normal distribution
    # with a mean of 0 and standard deviation of 2 * 10e-1
    dense_weights = RandomNormal(mean=0.0, stddev=(2 * 10e-1))
    
    # The biases for all layers are initialized from a normal distribution
    # with a mean of 0.5 and standard deviation of 10e-2
    biases = RandomNormal(mean=0.5, stddev=10e-2)
    
    def stem(input_shape):
        ''' Construct the Stem Group
            input_shape: input shape for input vector
        '''

        # entry convolutional layer and reduce feature maps by 75% (max pooling)
        model.add(Conv2D(64, (10, 10), activation='relu', kernel_initializer=conv_weights, bias_initializer=biases, input_shape=input_shape))
        model.add(MaxPooling2D((2, 2), strides=2))
        
    def block():
        ''' Construct a Convolutional Block '''
    
        # 2nd convolutional layer doubling the number of filters, and reduce feature maps by 75% (max pooling)
        model.add(Conv2D(128, (7, 7), activation='relu', kernel_initializer=conv_weights, bias_initializer=biases))
        model.add(MaxPooling2D((2, 2), strides=2))
    
        # 3rd convolutional layer and reduce feature maps by 75% (max pooling)
        model.add(Conv2D(128, (4, 4), activation='relu', kernel_initializer=conv_weights, bias_initializer=biases))
        model.add(MaxPooling2D((2, 2), strides=2))
        
        # 4th convolutional layer doubling the number of filters with no feature map downsampling
        model.add(Conv2D(256, (4, 4), activation='relu', kernel_initializer=conv_weights, bias_initializer=biases))
        # for a 105x105 input, the feature map size will be 6x6
        
    def encoder():
        ''' Construct the Encoding block '''
        # flatten the maps into a 1D vector
        model.add(Flatten())
    
        # use dense layer to produce a 4096 encoding of the flattened feature maps
        model.add(Dense(4096, activation='sigmoid', kernel_initializer=dense_weights, bias_initializer=biases))
        
    # Build the model
    stem(input_shape)
    block()
    encoder()
    return model

# Input shape for the Omniglot dataset
input_shape = (105, 105, 3)

# Create the twin model using the Sequential API  
model = twin(input_shape)

# Create input tensors for the left and right side (twins) of the network.
left_input  = Input(input_shape)
right_input = Input(input_shape)

# Create the encoders for the left and right side (twins)
left  = model( left_input )
right = model( right_input )

# Use Lambda method to create a custom layer for implementing a L1 distance layer.
L1Distance = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))

# Connect the left and right twins (via encoders) to the layer that calculates the
# distance between the encodings.
connected = L1Distance([left, right])

# Create the output layer for predicting the similarity from the distance layer
outputs = Dense(1,activation='sigmoid', kernel_initializer=dense_weights, bias_initializer=biases)(connected)
    
# Create the Siamese Network model
# Connect the left and right inputs to the outputs
model = Model(inputs=[left_input,right_input],outputs=outputs)

 
