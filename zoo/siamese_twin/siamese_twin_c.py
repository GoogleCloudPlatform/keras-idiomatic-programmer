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

# Siamese Net for One-Shot Image Classification (Koch, et. al.) + Composable
# Paper: https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pd

import tensorflow as tf
from tensorflow.keras import Input, Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Lambda
from tensorflow.keras.initializers import RandomNormal
import tensorflow.keras.backend as K

import sys
sys.path.append('../')
from models_c import Composable

class SiameseTwin(Composable):
    """ Construct a Siamese Twin network """
    global conv_weights, dense_weights, biases
   
    # The weights for the convolutional layers are initialized from a normal distribution
    # with a zero_mean and standard deviation of 10e-2
    conv_weights = RandomNormal(mean=0.0, stddev=10e-2)

    # The weights for the dense layers are initialized from a normal distribution
    # with a mean of 0 and standard deviation of 2 * 10e-1
    dense_weights = RandomNormal(mean=0.0, stddev=(2 * 10e-1))

    # The biases for all layers are initialized from a normal distribution
    # with a mean of 0.5 and standard deviation of 10e-2
    biases = RandomNormal(mean=0.5, stddev=10e-2)

    # Initial Hyperparameters
    hyperparameters = { 'initializer': 'glorot_uniform',
                        'regularizer': None,
                        'relu_clip'  : None,
                        'bn_epsilon' : None,
                        'use_bias'   : True
                      }

    def __init__(self, input_shape=(105, 105, 3),
                       **hyperparameters):
        """ Construct a Siamese Twin Neural Network 
            input_shape : input shape
            initializer : kernel initializer
            regularizer : kernel regularizer
            relu_clip   : max value for ReLU
            bn_epsilon  : epsilon for batch norm
            use_bias    : whether to use bias in conjunction with batch norm
        """
        # Configure the base (super) class
        Composable.__init__(self, input_shape, None, self.hyperparameters, **hyperparameters)
    
        # Build the twin model
        twin = self.twin(input_shape)

        # Create input tensors for the left and right side (twins) of the network.
        left_input  = Input(input_shape)
        right_input = Input(input_shape)

        # Create the encoders for the left and right side (twins)
        left  = twin( left_input )
        right = twin( right_input )

        # Use Lambda method to create a custom layer for implementing a L1 distance layer.
        L1Distance = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))

        # Connect the left and right twins (via encoders) to the layer that calculates the
        # distance between the encodings.
        connected = L1Distance([left, right])

        # Create the output layer for predicting the similarity from the distance layer
        outputs = self.Dense(connected, 1,activation='sigmoid', kernel_initializer=dense_weights, bias_initializer=biases)
    
	# Create the Siamese Network model
	# Connect the left and right inputs to the outputs
        self._model = Model(inputs=[left_input,right_input],outputs=outputs)

    def twin(self, input_shape):
        ''' Construct the model for both twins of the Siamese (connected) Network
            input_shape : input shape for input vector
        '''
    
        def stem(inputs):
            ''' Construct the Stem Group
                inputs: the input tensor
            '''

            # entry convolutional layer and reduce feature maps by 75% (max pooling)
            x = self.Conv2D(inputs, 64, (10, 10), activation='relu', kernel_initializer=conv_weights, bias_initializer=biases)
            x = MaxPooling2D((2, 2), strides=2)(x)
            return x
        
        def learner(x):
            ''' Construct the learner 
                x   : input to the learner
            '''
    
            # 2nd convolutional layer doubling the number of filters, and reduce feature maps by 75% (max pooling)
            x = self.Conv2D(x, 128, (7, 7), activation='relu', kernel_initializer=conv_weights, bias_initializer=biases)
            x = MaxPooling2D((2, 2), strides=2)(x)
    
            # 3rd convolutional layer and reduce feature maps by 75% (max pooling)
            x = self.Conv2D(x, 128, (4, 4), activation='relu', kernel_initializer=conv_weights, bias_initializer=biases)
            x = MaxPooling2D((2, 2), strides=2)(x)
        
            # 4th convolutional layer doubling the number of filters with no feature map downsampling
            x = self.Conv2D(x, 256, (4, 4), activation='relu', kernel_initializer=conv_weights, bias_initializer=biases)

            # for a 105x105 input, the feature map size will be 6x6
            return x
        
        def classifier(x):
            ''' Construct the classifier (Encoding block) 
                x  : input to the classifier
            '''

            # flatten the maps into a 1D vector
            x = Flatten()(x)
    
            # use dense layer to produce a 4096 encoding of the flattened feature maps
            x = self.Dense(x, 4096, activation='sigmoid', kernel_initializer=dense_weights, bias_initializer=biases)
            return x

        inputs = Input(shape=input_shape)
        x = stem(inputs)
        x = learner(x)
        outputs = classifier(x)

        return Model(inputs, outputs)
        

# Example
# siam = SiameseTwin()
