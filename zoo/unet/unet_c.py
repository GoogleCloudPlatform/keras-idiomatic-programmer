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

# U-Net
# Trainable params: 34,658,112
# Paper: https://arxiv.org/pdf/1505.04597.pdf 

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, ReLU, MaxPooling2D, Concatenate, Cropping2D
from tensorflow.keras.layers import Conv2DTranspose, Dropout
from tensorflow.keras.regularizers import l2

import sys
sys.path.append('../')
from models_c import Composable

class UNet(Composable):
    """ U-Net Convolutional Neural Network for Segmentation Tasks
    """
    groups = [ { 'n_filters': 64,  'crop': 88 },
               { 'n_filters': 128, 'crop': 40 },
               { 'n_filters': 256, 'crop': 16 },
               { 'n_filters': 512, 'crop': 4 }
             ]

    # Initial Hyperparameters
    hyperparameters = { 'initializer': 'he_normal',
                        'regularizer': None,
                        'relu_clip'  : None,
                        'bn_epsilon' : None,
                        'use_bias'   : False
                      }
    
    def __init__(self, groups=None,
                 input_shape=(572, 572, 3), n_classes=2, include_top=True,
                 **hyperparameters):
        """ Construct a U-Net Convolutiuonal Neural Network
	    groups      : contracting path groups
	    input_shape : input shape
    	    n_classes   : number of output classes
            include_top : whether to include classifier
            initializer : kernel initializer
            regularizer : kernel regularizer
            relu_clip   : max value for ReLU
            bn_epsilon  : epsilon for batch norm
            use_bias    : whether to use bias with batchnorm
        """
        # Configure the base (super) class
        Composable.__init__(self, input_shape, include_top, self.hyperparameters, **hyperparameters)

        # Predefined
        if groups is None:
            groups = self.groups

        # The input tensor
        inputs = Input(input_shape)

        # The stem convolutional group
        x = self.stem(inputs)

        # The learner
        outputs = self.learner(x, groups=groups)

        # The classifier 
        # Add hidden dropout for training-time regularization
        if include_top:
            outputs = self.classifier(outputs, n_classes, dropout=0.0)

        # Instantiate the Model
        self._model = Model(inputs, outputs)
        
    def stem(self, inputs):
        """ Construct the Stem Convolutional Group 
            inputs : the input vector
        """
        # no stem operation
        return inputs
    
    def learner(self, x, groups):
        """ Construct the Learner
            x     : input to the learner
            groups: contracting path groups
        """

        # Contracting Path
        x, e_groups = self.contracting(x, groups=groups)

        # Expansive Path
        x = self.expandsive(x, groups=e_groups)

        return x

    def contracting(self, x, **metaparameters):
        """ Construct Contracting Path (leftside)
            x      : input tensor to contracting path
            groups : contracting path groups
        """
        groups = metaparameters['groups']

        # Add each group on contracting path
        e_groups = [] # construct expansive groups
        for group in groups:
            n_filters = group['n_filters']
            crop      = group['crop']
            x, f = self.contract_group(x, n_filters, crop)
            # add parameters for corresponding expansive group
            e_groups.insert( 0, { 'n_filters': n_filters, 'fmap': f } )

        # Output from contracting path
        # Double the number of filters from the last path
        x = self.Conv2D(x, n_filters*2, (3, 3), strides=1, padding='valid')

        return x, e_groups

    def contract_group(self, x, n_filters, crop):
        """ Construct Contracting Group
            x        : input tensor to group
            n_filters: number of filters
            crop     : crop size for feature maps
        """
        # B(3, 3) convolutions
        x = self.Conv2D(x, n_filters, (3, 3), strides=1, padding='valid')
        x = self.ReLU(x)
        x = self.Conv2D(x, n_filters, (3, 3), strides=1, padding='valid')
        x = self.ReLU(x)
        # Crop the Feature Map
        f = Cropping2D(((crop, crop), (crop, crop)))(x)

        # Downsampling
        x = MaxPooling2D((2, 2), strides=2)(x)

        return x, f

    def expandsive(self, x, **metaparameters):
        """ Construct Expansive Path (rightside)
            x      : input tensor to expansive path
            groups : expansive  path groups
        """
        groups = metaparameters['groups']

        # Input to expansive path 
        n_filters = groups[0]['n_filters']
        x = self.Conv2D(x, n_filters, (3, 3), strides=1, padding='valid')

        # Add each group on expansive path
        for group in groups:
            n_filters = group['n_filters']
            fmap      = group['fmap']
            x = self.expand_group(x, fmap, n_filters)

        return x

    def expand_group(self, x, f, n_filters):
        """ Construct Expanding Group
            x        : input tensor to group
            f        : corresponding feature maps from contracting path
            n_filters: number of filters
        """
        # The Up Convolution (double feature map size)
        x = self.Conv2DTranspose(x, n_filters, (2, 2), strides=2)

        # Concatenate corresponding feature maps from contracting side
        x = Concatenate()([x, f])

        # Dimensionality Expansion
        x = self.Conv2D(x, n_filters*2, (3, 3), strides=1, padding='valid')
        x = self.ReLU(x)
        # Dimensionality Restoration
        x = self.Conv2D(x, n_filters, (3, 3), strides=1, padding='valid')
        x = self.ReLU(x)
        return x

    def classifier(self, x, n_classes, **metaparameters):
        """ Construct the classifier
            x        : input to the classifier
            n_classes: number of classes
            dropout  : percentage of dropout
        """
        dropout = metaparameters['dropout']

        if dropout > 0.0:
            x = Dropout(dropout)(x)

        x = self.Conv2D(x, n_classes, (1, 1), strides=1, padding='valid', activation='sigmoid') 
        return x


# Example U-Net
# unet = UNet()
