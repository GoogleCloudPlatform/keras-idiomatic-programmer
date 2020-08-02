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
import sys, os, json

class Training(object):
    ''' Training Class for Models '''

    def __init__(self):
        """ Constructor
        """
        pass

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

    def training(self, x_train=None, y_train=None, epochs=10, batch_size=32, lr=0.001, decay=(None, 0),
                 split=0.1, loss='categorical_crossentropy', metrics=['acc'], save=None):
        """ Full Training of the Model
            x_train    : training images
            y_train    : training labels
            epochs     : number of epochs
            batch_size : size of batch
            lr         : learning rate
            decay      : step-wise learning rate decay
            split      : percent to use as validation data
            loss       : loss function
            metrics    : metrics to report during training
            save       : where to store training metadata
        """
        print("\n*** Full Training")

        if x_train is None:
            x_train = self.x_train
            y_train = self.y_train

        t_epochs = 0

        if save is not None:
            for path in [ save, save + '/train']:
                try:
                    os.mkdir(path)
                except:
                    pass
            if os.path.isfile(save + '/train/chkpt.index'):
                self.model.load_weights(save + '/train/chkpt')
                print("\nIncemental Training, loaded previous weights")
                if os.path.isfile(save + '/train/train.json'):
                    with open(save + '/train/train.json', 'r') as f:
                        data = json.load(f)
                        t_epochs = data['epochs']
            elif os.path.isfile(save + '/pretext/chkpt.index'):
                self.model.load_weights(save + '/pretext/chkpt')
            elif os.path.isfile(save + '/tune/chkpt.index'):
                self.model.load_weights(save + '/tune/chkpt')
            elif os.path.isfile(save + '/warmup/chkpt.index'):
                self.model.load_weights(save + '/warmup/chkpt')
            elif os.path.isfile(save + '/init/chkpt.index'):
                self.model.load_weights(save + '/init/chkpt')

            if lr is None:
                if os.path.isfile(save + '/train/train.json'):
                    with open(save + '/train/train.json', 'r') as f:
                        data = json.load(f)
                        lr = data['lr']
                        batch_size = data['bs']
                elif os.path.isfile(save + '/tune/hp.json'):
                    with open(save + '/tune/hp.json', 'r') as f:
                        data = json.load(f)
                        lr = data['lr']
                        batch_size = data['bs']

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
            raise Exception("Training: invalid method for decay:", decay[0])

        if batch_size is None:
            batch_size = 32

        self.i_lr    = lr
        self.e_decay = decay
        self.e_steps = x_train.shape[0] // batch_size
        self.t_steps = self.e_steps * epochs
        self.compile(optimizer=Adam(lr=lr, decay=decay[1]), loss=loss, metrics=metrics)

        lrate = LearningRateScheduler(self.training_scheduler, verbose=1)
        result = self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=split, 
                       verbose=1, callbacks=[lrate])

        if save is not None:
            self.model.save_weights(save + '/train/chkpt')
            with open(save + '/train/train.json', 'w') as f:
                data = { 'lr': lr, 'bs': batch_size, 'epochs': t_epochs + epochs,
                         'val_loss': result.history['val_loss'], 'val_acc': result.history['val_acc'] }
                json.dump(data, f)

    def evaluate(self, x_test=None, y_test=None):
        """ Call underlying evaluate() method
            x_test     : training images
            y_test     : training labels
        """
        if x_test is None:
            x_test = self.x_test
            y_test = self.y_test

        print("\n*** Evaluation")
        return self._model.evaluate(x_test, y_test)
