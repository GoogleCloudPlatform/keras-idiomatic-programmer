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
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras import layers
from tensorflow.keras.layers import ReLU, Dense, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import DepthwiseConv2D, SeparableConv2D, Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D, Activation, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.compat.v1.keras.initializers import glorot_uniform, he_normal
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import tensorflow_datasets as tfds
import tensorflow.keras.backend as K
import numpy as np
from sklearn.model_selection import train_test_split

import random
import math
import sys

from layers_c import Layers
from preprocess_c import Preprocess
from pretraining_c import Pretraining
from hypertune_c import HyperTune

class Composable(Layers, Preprocess, Pretraining, HyperTune):
    ''' Composable base (super) class for Models '''

    def __init__(self, init_weights=None, reg=None, relu=None, bias=True):
        """ Constructor
            init_weights : kernel initializer
            reg          : kernel regularizer
            relu         : clip value for ReLU
            bias         : whether to use bias
        """
        Layers.__init__(self, init_weights, reg, relu, bias)
        Preprocess.__init__(self)
        Pretraining.__init__(self)
        HyperTune.__init__(self)

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

    ###
    # Training
    ###

    def compile(self, loss='categorical_crossentropy', optimizer=Adam(lr=0.001, decay=1e-5), metrics=['acc']):
        """ Compile the model for training
            loss     : the loss function
            optimizer: the optimizer
            metrics  : metrics to report
        """
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    # training variables
    hidden_dropout = None # hidden dropout in classifier
    i_lr           = 0    # initial rate during full training
    e_decay        = 0    # weight decay rate during full training
    e_steps        = 0    # number of steps (batches) in an epoch
    t_steps        = 0    # total number of steps in training job

    def time_decay(self, epoch, lr):
        """ Time-based Decay
        """
        return lr * (1. / (1. + self.e_decay[1] * epoch))

    def step_decay(self, epoch, lr):
        """ Step-based (polynomial) decay
        """
        return self.i_lr * self.e_decay[1]**(epoch)

    def exp_decay(self, epoch, lr):
        """ Exponential Decay
        """
        return self.i_lr * math.exp(-self.e_decay[1] * epoch)

    def cosine_decay(self, epoch, lr, alpha=0.0):
        """ Cosine Decay
        """
        cosine_decay = 0.5 * (1 + np.cos(np.pi * (self.e_steps * epoch) / self.t_steps))
        decayed = (1 - alpha) * cosine_decay + alpha
        return lr * decayed

    def training_scheduler(self, epoch, lr):
        """ Learning Rate scheduler for full-training
            epoch : epoch number
            lr    : current learning rate
        """
        # First epoch (not started) - do nothing
        if epoch == 0:
            return lr

        # Hidden dropout unit in classifier
        if self.hidden_dropout is not None:
            # If training accuracy and validation accuracy more than 3% apart
            if self.model.history.history['acc'][epoch-1] > self.model.history.history['val_acc'][epoch-1] + 0.03:
                if self.hidden_dropout.rate == 0.0:
                    self.hidden_dropout.rate = 0.5
                elif self.hidden_dropout.rate < 0.75:
                    self.hidden_dropout.rate *= 1.1
                print("*** Overfitting, set dropout to", self.hidden_dropout.rate)
            else:
                if self.hidden_dropout.rate != 0.0:
                    print("*** Turning off dropout")
                    self.hidden_dropout.rate = 0.0

        if self.e_decay[0] is None:
            return lr

        # Decay the learning rate
        if self.e_decay[0] == 'time':
            lr = self.time_decay(epoch, lr)
        elif self.e_decay[0] == 'step':
            lr = self.step_decay(epoch, lr)
        elif self.e_decay[0] == 'exp':
            lr = self.exp_decay(epoch, lr)
        else:
            lr = self.cosine_decay(epoch, lr)
        return lr

    def training(self, x_train, y_train, epochs=10, batch_size=32, lr=0.001, decay=(None, 0),
                 split=0.1):
        """ Full Training of the Model
            x_train    : training images
            y_train    : training labels
            epochs     : number of epochs
            batch_size : size of batch
            lr         : learning rate
            decay      : step-wise learning rate decay
            split      : percent to use as validation data
        """

        print("*** Full Training")

        # Check for hidden dropout layer in classifier
        for layer in self.model.layers:
            if isinstance(layer, Dropout):
                self.hidden_dropout = layer
                break    

        if decay is None or 0:
            decay = (None, 0)
        elif isinstance(decay, float):
            decay = ('time', decay)
        elif not isinstance(decay, tuple):
            raise Exception("Training: decay must be (time, value)")
        elif decay[0] not in [None, 'time', 'step', 'exp', 'cosine']:
            raise Exception("Training: invalid method for decay")

        self.i_lr    = lr
        self.e_decay = decay
        self.e_steps = x_train.shape[0] // batch_size
        self.t_steps = self.e_steps * epochs
        self.compile(optimizer=Adam(lr=lr, decay=decay[1]))

        lrate = LearningRateScheduler(self.training_scheduler, verbose=1)
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=split, verbose=1,
                       callbacks=[lrate])

    def evaluate(self, x_test, y_test):
        """ Call underlying evaluate() method
        """
        return self._model.evaluate(x_test, y_test)

    def cifar10(self, epochs=10, decay=('cosine', 0)):
        """ Train on CIFAR-10
            epochs : number of epochs for full training
        """
        from tensorflow.keras.datasets import cifar10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train, x_test = self.standardization(x_train, x_test)
        y_train = to_categorical(y_train, 10)
        y_test  = to_categorical(y_test, 10)
        y_train = self.label_smoothing(y_train, 10, 0.1)

        # compile the model
        self.compile(loss='categorical_crossentropy', metrics=['acc'])

        self.warmup(x_train, y_train)

        lr, batch_size = self.random_search(x_train, y_train, x_test, y_test)

        self.training(x_train, y_train, epochs=epochs, batch_size=batch_size,
                      lr=lr, decay=decay)
        self.evaluate(x_test, y_test)

    def cifar100(self, epochs=20, decay=('cosine', 0)):
        """ Train on CIFAR-100
            epochs : number of epochs for full training
        """
        from tensorflow.keras.datasets import cifar100
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        x_train, x_test = self.normalization(x_train, x_test)
        y_train = to_categorical(y_train, 100)
        y_test  = to_categorical(y_test, 100)
        y_train = self.label_smoothing(y_train, 10, 0.1)
        self.compile(loss='categorical_crossentropy', metrics=['acc'])

        self.warmup(x_train, y_train)

        lr, batch_size = self.grid_search(x_train, y_train, x_test, y_test)

        self.training(x_train, y_train, epochs=epochs, batch_size=batch_size,
                      lr=lr, decay=decay)
        self.evaluate(x_test, y_test)

    def coil100(self, epochs=20, decay=('cosine', 0)):
        """
        """
        # Get TF.dataset generator for COIL100
        train, info = tfds.load('coil100', split='train', shuffle_files=True, with_info=True, as_supervised=True)
        n_classes = info.features['label'].num_classes
        n_images = info.splits['train'].num_examples
        input_shape = info.features['image'].shape

        # Get the dataset into memory
        train = train.shuffle(n_images).batch(n_images)
        for images, labels in train.take(1):
            pass
    
        images = np.asarray(images)
        images, _ = self.standardization(images, None)
        labels = to_categorical(np.asarray(labels), n_classes)

        # split the dataset into train/test
        x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2)

        self.compile(loss='categorical_crossentropy', metrics=['acc'])

        self.warmup(x_train, y_train)

        lr, batch_size = self.grid_search(x_train, y_train, x_test, y_test)

        self.training(x_train, y_train, epochs=epochs, batch_size=batch_size,
                      lr=lr, decay=decay)
        self.evaluate(x_test, y_test)

