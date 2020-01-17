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
from tensorflow.keras.layers import ReLU, Dense, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import DepthwiseConv2D, SeparableConv2D, Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D, Activation, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.backend as K

class Composable(object):
    ''' Composable base (super) class for Models '''
    init_weights = 'he_normal'	# weight initialization
    reg          = None         # kernel regularizer
    relu         = None         # ReLU max value

    def __init__(self, init_weights=None, reg=None, relu=None):
        """ Constructor
            init_weights : kernel initializer
            relu         :
            reg          : kernel regularizer
        """
        if init_weights is not None:
            self.init_weights = init_weights
        if reg is not None:
            self.reg = reg
        if relu is not None:
            self.relu = relu

        # Feature maps encoding at the bottleneck layer in classifier (high dimensionality)
        self._encoding = None
        # Pooled and flattened encodings at the bottleneck layer (low dimensionality)
        self._embedding = None
        # Pre-activation conditional probabilities for classifier
        self._probabilities = None
        # Post-activation conditional probabilities for classifier
        self._softmax = None

        self._model = None

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, _model):
        self._model = _model

    @property
    def encoding(self):
        return self._encoding

    @encoding.setter
    def encoding(self, layer):
        self._encoding = layer

    @property
    def embedding(self):
        return self._embedding

    @embedding.setter
    def embedding(self, layer):
        self._embedding = layer

    @property
    def probabilities(self):
        return self._probabilities

    @probabilities.setter
    def probabilities(self, layer):
        self._probabilities = layer

    def classifier(self, x, n_classes, **metaparameters):
      """ Construct the Classifier Group 
          x         : input to the classifier
          n_classes : number of output classes
          pooling   : type of feature map pooling
      """
      if 'pooling' in metaparameters:
          pooling = metaparameters['pooling']
      else:
          pooling = GlobalAveragePooling2D
      if 'dropout' in metaparameters:
          dropout = metaparameters['dropout']
      else:
          dropout = None

      if pooling is not None:
          # Save the encoding layer (high dimensionality)
          self.encoding = x

          # Pooling at the end of all the convolutional groups
          x = pooling()(x)

          # Save the embedding layer (low dimensionality)
          self.embedding = x

      if dropout is not None:
          x = Dropout(dropout)(x)

      # Final Dense Outputting Layer for the outputs
      x = self.Dense(x, n_classes, **metaparameters)
      
      # Save the pre-activation probabilities layer
      self.probabilities = x
      outputs = Activation('softmax')(x)
      # Save the post-activation probabilities layer
      self.softmax = outputs
      return outputs

    def Dense(self, x, units, activation=None, use_bias=True, **hyperparameters):
        """ Construct Dense Layer
            x           : input to layer
            activation  : activation function
            use_bias    : whether to include the bias
            init_weights: kernel initializer
            reg         : kernel regularizer
        """
        if 'reg' in hyperparameters:
            reg = hyperparameters['reg']
        else:
            reg = self.reg
        if 'init_weights' in hyperparameters:
            init_weights = hyperparameters['init_weights']
        else:
            init_weights = self.init_weights
            
        x = Dense(units, activation, use_bias=use_bias,
                  kernel_initializer=init_weights, kernel_regularizer=reg)(x)
        return x

    def Conv2D(self, x, n_filters, kernel_size, strides=(1, 1), padding='valid', activation=None, use_bias=True, **hyperparameters):
        """ Construct a Conv2D layer
            x           : input to layer
            n_filters   : number of filters
            kernel_size : kernel (filter) size
            strides     : strides
            padding     : how to pad when filter overlaps the edge
            activation  : activation function
            use_bias    : whether to include the bias
            init_weights: kernel initializer
            reg         : kernel regularizer
        """
        if 'reg' in hyperparameters:
            reg = hyperparameters['reg']
        else:
            reg = self.reg
        if 'init_weights' in hyperparameters:
            init_weights = hyperparameters['init_weights']
        else:
            init_weights = self.init_weights

        x = Conv2D(n_filters, kernel_size, strides=strides, padding=padding, activation=activation, use_bias=use_bias,
                   kernel_initializer=init_weights, kernel_regularizer=reg)(x)
        return x

    def Conv2DTranspose(self, x, n_filters, kernel_size, strides=(1, 1), padding='valid', activation=None, use_bias=True, **hyperparameters):
        """ Construct a Conv2DTranspose layer
            x           : input to layer
            n_filters   : number of filters
            kernel_size : kernel (filter) size
            strides     : strides
            padding     : how to pad when filter overlaps the edge
            activation  : activation function
            use_bias    : whether to include the bias
            init_weights: kernel initializer
            reg         : kernel regularizer
        """
        if 'reg' in hyperparameters:
            reg = hyperparameters['reg']
        else:
            reg = self.reg
        if 'init_weights' in hyperparameters:
            init_weights = hyperparameters['init_weights']
        else:
            init_weights = self.init_weights

        x = Conv2DTranspose(n_filters, kernel_size, strides=strides, padding=padding, activation=activation, use_bias=use_bias,
                            kernel_initializer=init_weights, kernel_regularizer=reg)(x)
        return x

    def DepthwiseConv2D(self, x, kernel_size, strides=(1, 1), padding='valid', activation=None, use_bias=True, **hyperparameters):
        """ Construct a DepthwiseConv2D layer
            x           : input to layer
            kernel_size : kernel (filter) size
            strides     : strides
            padding     : how to pad when filter overlaps the edge
            activation  : activation function
            use_bias    : whether to include the bias
            init_weights: kernel initializer
            reg         : kernel regularizer
        """
        if 'reg' in hyperparameters:
            reg = hyperparameters['reg']
        else:
            reg = self.reg
        if 'init_weights' in hyperparameters:
            init_weights = hyperparameters['init_weights']
        else:
            init_weights = self.init_weights

        x = DepthwiseConv2D(kernel_size, strides=strides, padding=padding, activation=activation, use_bias=use_bias,
                            kernel_initializer=init_weights, kernel_regularizer=reg)(x)
        return x

    def SeparableConv2D(self, x, n_filters, kernel_size, strides=(1, 1), padding='valid', activation=None, use_bias=True, **hyperparameters):
        """ Construct a SeparableConv2D layer
            x           : input to layer
            n_filters   : number of filters
            kernel_size : kernel (filter) size
            strides     : strides
            padding     : how to pad when filter overlaps the edge
            activation  : activation function
            use_bias    : whether to include the bias
            init_weights: kernel initializer
            reg         : kernel regularizer
        """
        if 'reg' in hyperparameters:
            reg = hyperparameters['reg']
        else:
            reg = self.reg
        if 'init_weights' in hyperparameters:
            init_weights = hyperparameters['init_weights']
        else:
            init_weights = self.init_weights

        x = SeparableConv2D(n_filters, kernel_size, strides=strides, padding=padding, activation=activation, use_bias=use_bias,
                            kernel_initializer=init_weights, kernel_regularizer=reg)(x)

        return x

    def ReLU(self, x):
        """ Construct ReLU activation function
            x  : input to activation function
        """
        x = ReLU(self.relu)(x)
        return x
	
    def HS(self, x):
        """ Construct Hard Swish activation function
            x  : input to activation function
        """
        return (x * K.relu(x + 3, max_value=6.0)) / 6.0

    def BatchNormalization(self, x, **params):
        """ Construct a Batch Normalization function
            x : input to function
        """
        x = BatchNormalization(epsilon=1.001e-5, **params)(x)
        return x
        

    def compile(self, loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001, decay=1e-5), metrics=['acc']):
        """ Compile the model for training
            loss     : the loss function
            optimizer: the optimizer
            metrics  : metrics to report
        """
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    # training variables
    hidden_dropout = None # hidden dropout in classifier
    w_lr           = 0    # target warmup rate
    w_epochs       = 0    # number of epochs in warmup

    def warmup_scheduler(self, epoch, lr):
        """ learning rate schedular for warmup training
            epoch : current epoch iteration
            lr    : current learning rate
        """
        if epoch == 0:
           return lr
        return epoch * self.w_lr / self.w_epochs

    def warmup(self, x_train, y_train, epochs=5, loss='sparse_categorical_crossentropy', 
               s_lr=1e-6, e_lr=0.001):
        """ Warmup for numerical stability
            x_train : training images
            y_train : training labels
            epochs  : number of epochs for warmup
            loss    : the loss function
            s_lr    : start warmup learning rate
            e_lr    : end warmup learning rate
        """
        # Setup learning rate scheduler
        self.compile(loss=loss, optimizer=Adam(s_lr), metrics=['acc'])
        lrate = LearningRateScheduler(self.warmup_scheduler, verbose=1)
        self.w_epochs = epochs
        self.w_lr     = e_lr

        # Train the model
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=32, verbose=1,
                       callbacks=[lrate])

    def grid_search(self, x_train, y_train, x_test, y_test, epochs=3, steps=250, loss='sparse_categorical_crossentropy',
                          lr_range=[0.0001, 0.001, 0.01, 0.1], batch_range=[32, 128]):
        """ Do a grid search for hyperparameters
            x_train : training images
            y_train : training labels
            epochs  : number of epochs
            steps   : number of steps per epoch
            loss    : loss function
            lr_range: range for searching learning rate
            batch_range: range for searching batch size
        """
        # Save the original weights
        weights = self.model.get_weights()

        # Create generator for training in steps
        datagen = ImageDataGenerator()

        # Search learning rate
        v_loss = []
        for lr in lr_range:
            # Compile the model for the new learning rate
            self.compile(loss=loss, optimizer=Adam(lr))
            
            # Train the model
            print("Learning Rate", lr)
            self.model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_range[0]),
                                     epochs=epochs, steps_per_epoch=steps, verbose=1)

            # Evaluate the model
            result = self.model.evaluate(x_test, y_test)
            v_loss.append(result[0])
            
            # Reset the weights
            self.model.set_weights(weights)

        # Find the best starting learning rate based on validation loss
        best = 9999.0
        for _ in range(len(lr_range)):
            if v_loss[_] < best:
                best = v_loss[_]
                lr = lr_range[_]

        print("Selected best learning rate:", lr)

        # Compile the model for the new learning rate
        self.compile(loss=loss, optimizer=Adam(lr))
        
        v_loss = []
        for bs in batch_range:
            print("Batch Size", bs)

            # equalize the number of examples per epoch
            steps = int(batch_range[0] * steps / bs)

            self.model.fit_generator(datagen.flow(x_train, y_train, batch_size=bs),
                                     epochs=epochs, steps_per_epoch=steps, verbose=1)

            # Evaluate the model
            result = self.model.evaluate(x_test, y_test)
            v_loss.append(result[0])
            
            # Reset the weights
            self.model.set_weights(weights)

        # Find the best batch size based on validation loss
        best = 9999.0
        for _ in range(len(batch_range)):
            if v_loss[_] < best:
                best = v_loss[_]
                bs = batch_range[_]

        print("Selected best batch size:", bs)

        # return the best learning rate and batch size
        return lr, bs

    def training_schedule(self, epoch, lr):
        """
        """
        print("ACC", self.model.history.history['acc'], self.model.history.history['val_acc'])
        if self.model.history.history['acc'] > self.model.history.history['val_acc'] + 3:
            hidden_dropout.rate = 0.5
            print("Overfitting, adding 50% dropout")
        else:
            hidden_dropout.rate = 0.0
        return lr

    def training(self, x_train, y_train, epochs=10, batch_size=32):
        """ Full Training of the Model
            x_train    : training images
            y_train    : training labels
            epochs     : number of epochs
            batch_size : size of batch
        """

        # Check for hidden dropout layer in classifier
        for layer in self.model.layers:
            if isinstance(layer, Dropout):
                hidden_dropout = layer
                break    

        lrate = LearningRateScheduler(self.trainig_scheduler, verbose=1)
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1,
                       callbacks=[lrate])

    def cifar10(self, epochs=10):
        """ Train on CIFAR-10
            epochs : number of epochs for full training
        """
        from tensorflow.keras.datasets import cifar10
        import numpy as np
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = (x_train / 255.0).astype(np.float32)
        x_test  = (x_test  / 255.0).astype(np.float32)

        print("Warmup the model for numerical stability")
        self.warmup(x_train, y_train)

        print("Hyperparameter search")
        lr, batch_size = self.grid_search(x_train, y_train, x_test, y_test)

        print("Full training")
        self.compile(optimizer=Adam(lr=lr, decay=1e-5))
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=32, validation_split=0.1, verbose=1)
        self.model.evaluate(x_test, y_test)

    def cifar100(self, epochs=20):
        """ Train on CIFAR-100
            epochs : number of epochs for full training
        """
        from tensorflow.keras.datasets import cifar100
        import numpy as np
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        x_train = (x_train / 255.0).astype(np.float32)
        x_test  = (x_test  / 255.0).astype(np.float32)

        print("Warmup the model for numerical stability")
        self.warmup(x_train, y_train)

        print("Full training")
        self.compile()
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=32, validation_split=0.1, verbose=1)
        self.model.evaluate(x_test, y_test)
