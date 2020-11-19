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

# JumpNet (50, 101, 152)
# Residual Groups and Blocks are ResNet v2 - w/o projection block
# Stem convolution is a stack of two 3x3 filters (factorized 5x5), as in Inception v3

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, ReLU
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Add, Concatenate
from tensorflow.keras.regularizers import l2

import sys, json
sys.path.append('../')
from models_c import Composable

class JumpNet(Composable):
    """ Jump Convolutional Neural Network 
    """
    # Meta-parameter: list of groups: number of filters and number of blocks
    groups = { 50 : [ { 'n_filters' : 64, 'n_blocks': 3 },
                      { 'n_filters': 128, 'n_blocks': 4 },
                      { 'n_filters': 256, 'n_blocks': 6 },
                      { 'n_filters': 512, 'n_blocks': 3 } ],            # ResNet50
               101: [ { 'n_filters' : 64, 'n_blocks': 3 },
                      { 'n_filters': 128, 'n_blocks': 4 },
                      { 'n_filters': 256, 'n_blocks': 23 },
                      { 'n_filters': 512, 'n_blocks': 3 } ],            # ResNet101
               152: [ { 'n_filters' : 64, 'n_blocks': 3 },
                      { 'n_filters': 128, 'n_blocks': 8 },
                      { 'n_filters': 256, 'n_blocks': 36 },
                      { 'n_filters': 512, 'n_blocks': 3 } ]             # ResNet152
             }

    # Initial Hyperparameters
    hyperparameters = { 'initializer': 'he_normal',
                        'regularizer': l2(0.001),
                        'relu_clip'  : None,
                        'bn_epsilon' : None,
                        'use_bias'   : False
                      }

    def __init__(self, n_layers, stem={ 'n_filters': [32, 64], 'pooling': 'feature' },
                 input_shape=(224, 224, 3), n_classes=1000, include_top=True,
                 **hyperparameters):
        """ Construct a Jump Convolutional Neural Network 
            n_layers    : number of layers
            stem        : number of filters in the stem convolutional stack
            input_shape : input shape
            n_classes   : number of output classes
            include_top : whether to include classifier
            regulalizer : kernel regularizer
            relu_clip   : max value for ReLU
            initializer : kernel initializer
            bn_epsilon  : epsilon for batch norm
            use_bias    : whether to use bias with batchnorm
        """
        # Configure the base (super) class
        Composable.__init__(self, input_shape, include_top,
                            self.hyperparameters, **hyperparameters)

        # predefined
        if isinstance(n_layers, int):
            if n_layers not in [50, 101, 152]:
                raise Exception("JumpNet: Invalid value for n_layers")
            groups = list(self.groups[n_layers])
        # user defined
        else:
            groups = n_layers

        # The input tensor
        inputs = Input(input_shape)

        # The stem convolutional group
        x = self.stem(inputs, stem=stem)

        # The learner
        outputs = self.learner(x, groups=groups)

        # The classifier
        # Add hidden dropout for training-time regularization
        if include_top:
            outputs = self.classifier(outputs, n_classes)

        # Instantiate the Model
        self._model = Model(inputs, outputs)

    def stem(self, inputs, **metaparameters):
        """ Construct the Stem Convolutional Group 
            inputs : the input vector
            stack  : convolutional filters in two 3x3 stack
        """
        stack = metaparameters['stem']
        n_filters = stack['n_filters']

        pooling = stack['pooling']
        if pooling == 'feature':
           strides = (2, 2)
        else:
           strides = (1, 1)
    
        # Stack of two 3x3 filters
        x = self.Conv2D(inputs, n_filters[0], (3, 3), strides=strides, padding='same')
        x = self.BatchNormalization(x)
        x = self.ReLU(x)

        x = self.Conv2D(x, n_filters[1], (3, 3), strides=(1, 1), padding='same')
        x = self.BatchNormalization(x)
        x = self.ReLU(x)

        return x

    def learner(self, x, **metaparameters):
        """ Construct the Learner
            x     : input to the learner
            groups: list of groups: number of filters and blocks
        """
        groups = metaparameters['groups']

        # Residual Groups
        for group in groups:
            x = self.group(x, **group)
        return x
    
    def group(self, x, **metaparameters):
        """ Construct a Residual Group
            x         : input into the group
            n_blocks  : number of residual blocks with identity link
            n_filters : number of filters per group
        """
        n_blocks  = metaparameters['n_blocks']
        n_filters = metaparameters['n_filters']

        # Save the input to the group for the jump link at the end.
        shortcut = self.BatchNormalization(x)
        shortcut = self.Conv2D(shortcut, n_filters, (1, 1), strides=(2, 2))
    
        # Identity residual blocks
        for _ in range(n_blocks):
            x = self.identity_block(x, **metaparameters)

        # Feature Pooling at the end of the group
        x = self.BatchNormalization(x)
        x = self.ReLU(x)
        x = self.Conv2D(x, n_filters, (1, 1), strides=(2, 2))

        # Construct the jumpn link
        x = Concatenate()([shortcut, x])
        return x

    def identity_block(self, x, **metaparameters):
        """ Construct a Bottleneck Residual Block with Identity Link
            x        : input into the block
            n_filters: number of filters
        """
        n_filters = metaparameters['n_filters']
    
        # Save input vector (feature maps) for the identity link
        shortcut = x
    
        ## Construct the 1x1, 3x3, 1x1 convolution block
    
        # Dimensionality reduction
        x = self.BatchNormalization(x)
        x = self.ReLU(x)
        x = self.Conv2D(x, n_filters, (1, 1), strides=(1, 1))

        # Bottleneck layer
        x = self.BatchNormalization(x)
        x = self.ReLU(x)
        x = self.Conv2D(x, n_filters, (3, 3), strides=(1, 1), padding="same")

        # no dimensionality restoration
        x = self.BatchNormalization(x)
        x = self.ReLU(x)
        x = self.Conv2D(x, n_filters, (1, 1), strides=(1, 1))

        # Add the identity link (input) to the output of the residual block
        x = Add()([shortcut, x])
        return x

# Example of JumpNet  for CIFAR-10
groups = [ { 'n_filters': 32, 'n_blocks': 3 },
           { 'n_filters': 64,  'n_blocks': 4 },
           { 'n_filters': 128, 'n_blocks': 3 }]
jumpnet = JumpNet(n_layers=groups, stem={ 'n_filters': [16, 32], 'pooling': None }, 
                  input_shape=(32, 32, 3), n_classes=10,
                  regularizer=l2(0.005))
jumpnet.summary()

if __name__ == '__main__':
    import sys
    from tensorflow.keras.datasets import cifar10
    from tensorflow.keras.utils import to_categorical
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    jumpnet.load_data((x_train, y_train), (x_test, y_test), std=True, onehot=True, smoothing=0.1)

    # compile the model
    jumpnet.compile(loss='categorical_crossentropy', metrics=['acc'])

    lr = None
    bs = None
    for _ in range(1, len(sys.argv)):
        if sys.argv[_].startswith('lr'):
           # learning rate
           lr = float(sys.argv[_].split('=')[1])
        elif sys.argv[_].startswith('bs'):
           # batch size
           bs = int(sys.argv[_].split('=')[1])
        elif sys.argv[_].startswith('init'):
           # Use Lottery ticket approach for best initialization draw
           value = sys.argv[_].split('=')[1].split(',')
           ndraws = int(value[0])
           if len(value) > 1:
               early = bool(value[1])
           else:
               early = False
           jumpnet.init_draw(ndraws=ndraws, early=early, save='cifar10')

        elif sys.argv[_].startswith('warmup'):
           # Warmup the weight distribution for numeric stability
           epochs = int(sys.argv[_].split('=')[1])
           jumpnet.warmup(epochs=epochs, save='cifar10')

        elif sys.argv[_].startswith("tune"):
           # Do hyperparameter tuning from warmup
           trials = int(sys.argv[_].split('=')[1])
           lr, bs = jumpnet.random_search(trials=trials, save='cifar10')

        elif sys.argv[_].startswith('pretext'):
           # Do pretext task
           zigsaw = int(sys.argv[_].split('=')[1])
           jumpnet.pretext(zigsaw=zigsaw, lr=lr, batch_size=bs, save='cifar10')

        elif sys.argv[_].startswith('train'):
           # Do full training
           epochs = int(sys.argv[_].split('=')[1])
           jumpnet.training(epochs=epochs, batch_size=bs, lr=lr, decay=('cosine', 0), save='cifar10')
           jumpnet.evaluate()
           
