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

class HyperTune(object):
    ''' Hyperparameter tuning  base (super) class for Composable Models '''

    def __init__(self):
        """ Constructor
        """
        pass

    ###
    # Hyperparameter Tuning
    ###

    def _tune(self, x_train, y_train, x_test, y_test, epochs, steps, lr, batch_size, weights, loss, metrics):
        """ Helper function for hyperparameter tuning
            x_train   : training images
            y_train   : training labels
            x_test    : test images
            y_test    : test labels
            lr        : trial learning rate
            batch_size: the batch size (constant)
            epochs    : the number of epochs
            steps     : steps per epoch
            weights   : warmup weights
            loss      : loss function
            metrics   : metrics to report during training
        """
        # Compile the model for the new learning rate
        self.compile(optimizer=Adam(lr), loss=loss, metrics=metrics)

        # Create generator for training in steps
        datagen = ImageDataGenerator()
         
        # Train the model
        print("\n*** Learning Rate", lr, "Batch Size", batch_size)
        self.model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
                                 epochs=epochs, steps_per_epoch=steps, verbose=1)

        # Evaluate the model
        result = self.evaluate(x_test, y_test)
         
        # Reset the weights
        self.model.set_weights(weights)

        return result

    def grid_search(self, x_train=None, y_train=None, x_test=None, y_test=None, epochs=3, steps=250,
                          lr_range=[0.0001, 0.001, 0.01, 0.1], batch_range=[32, 128],
                          loss='categorical_crossentropy', metrics=['acc'], save=None):
        """ Do a grid search for hyperparameters
            x_train : training images
            y_train : training labels
            epochs  : number of epochs
            steps   : number of steps per epoch
            lr_range: range for searching learning rate
            batch_range: range for searching batch size
            loss    : loss function
            metrics : metrics to report during training
        """
        if x_train is None:
            x_train = self.x_train
            y_train = self.y_train
            x_test  = self.x_test
            y_test  = self.y_test

        if save is not None:
            for path in [ save, save + '/tune']:
                try:
                    os.mkdir(path)
                except:
                    pass
            if os.path.isfile(save + '/warmup/chkpt.index'):
                self.model.load_weights(save + '/warmup/chkpt')
            elif os.path.isfile(save + '/init/chkpt.index'):
                self.model.load_weights(save + '/warmup/chkpt')

        print("\n*** Hyperparameter Grid Search")

        # Save the original weights
        weights = self.model.get_weights()

        # Search learning rate
        v_loss = []
        for lr in lr_range:
            result = self._tune(x_train, y_train, x_test, y_test, epochs, steps, lr, batch_range[0], weights, loss, metrics)
            v_loss.append(result[0])
            
        # Find the best starting learning rate based on validation loss
        best = sys.float_info.max
        for _ in range(len(lr_range)):
            if v_loss[_] < best:
                best = v_loss[_]
                lr = lr_range[_]

        # Best was smallest learning rate
        if lr == lr_range[0]:
            # try 1/2 the lowest learning rate
            result = self._tune(x_train, y_train, x_test, y_test, epochs, steps, (lr / 2.0), batch_range[0], weights, losss, metrics)

            # 1/2 of lr is even better
            if result[0] < best:
                lr = lr / 2.0
            # try halfway between the first and second value
            else:
                n_lr = (lr_range[0] + lr_range[1]) / 2.0
                result = self._tune(x_train, y_train, x_test, y_test, epochs, steps, n_lr, batch_range[0], weights, loss, metrics)

                # 1/2 of lr is even better
                if result[0] < best:
                    lr = lr / 2.0
                
        elif lr == lr_range[len(lr_range)-1]:
            # try 2X the largest learning rate
            result = self._tune(x_train, y_train, x_test, y_test, epochs, steps, (lr * 2.0), batch_range[0], weights, loss, metrics)

            # 2X of lr is even better
            if result[0] < best:
                lr = lr * 2.0
		
        print("*** Selected best learning rate:", lr)

        # Compile the model for the new learning rate
        self.compile(optimizer=Adam(lr), loss=loss, metrics=metrics)
        
        v_loss = []
        # skip the first batch size - since we used it in searching learning rate
        datagen = ImageDataGenerator()
        for bs in batch_range[1:]:
            print("*** Batch Size", bs)

            # equalize the number of examples per epoch
            steps = int(batch_range[0] * steps / bs)

            self.model.fit(datagen.flow(x_train, y_train, batch_size=bs),
                                     epochs=epochs, steps_per_epoch=steps, verbose=1)

            # Evaluate the model
            result = self.evaluate(x_test, y_test)
            v_loss.append(result[0])
            
            # Reset the weights
            self.model.set_weights(weights)

        # Find the best batch size based on validation loss
        best = sys.float_info.max
        bs = batch_range[0]
        for _ in range(len(batch_range)-1):
            if v_loss[_] < best:
                best = v_loss[_]
                bs = batch_range[_]

        print("*** Selected best batch size:", bs)

        if save is not None:
            with open(save + '/tune/hp.json', 'w') as f:
                data = { 'lr' : lr, 'bs': bs }
                json.dump(data, f)
            self.model.save_weights(save + '/tune/chkpt')

        # return the best learning rate and batch size
        return lr, bs

    def random_search(self, x_train=None, y_train=None, x_test=None, y_test=None, epochs=3, steps=250,
                          lr_range=[0.0001, 0.001, 0.01, 0.1], batch_range=[32, 128], 
                          loss='categorical_crossentropy', metrics=['acc'], trials=5, save=None):
        """ Do a grid search for hyperparameters
            x_train : training images
            y_train : training labels
            epochs  : number of epochs
            steps   : number of steps per epoch
            lr_range: range for searching learning rate
            batch_range: range for searching batch size
            loss    : loss function
            metrics : metrics to report during training
            trials  : maximum number of trials
        """
        if x_train is None:
            x_train = self.x_train
            y_train = self.y_train
            x_test  = self.x_test
            y_test  = self.y_test

        if save is not None:
            for path in [ save, save + '/tune']:
                try:
                    os.mkdir(path)
                except:
                    pass
            if os.path.isfile(save + '/warmup/chkpt.index'):
                self.model.load_weights(save + '/warmup/chkpt')
            elif os.path.isfile(save + '/init/chkpt.index'):
                self.model.load_weights(save + '/warmup/chkpt')

        print("\n*** Hyperparameter Random Search")

        # Save the original weights
        weights = self.model.get_weights()

        # Base the number of steps on the min batch size to try
        min_bs = np.min(batch_range)

        best = (0, 0, 0)

        # lr values already tried, as not to repeat
        tried = []
        for _ in range(trials):
            print("\nTrial ", _ + 1, "of", trials)

            lr = lr_range[random.randint(0, len(lr_range)-1)]
            bs = batch_range[random.randint(0, len(batch_range)-1)]

            # Check for repeat
            if (lr, bs) in tried:
                print("Random Selection already tried", (lr, bs))
                continue
            tried.append( (lr, bs))

            # Adjust steps so each trial sees same number of examples
            trial_steps = int(min_bs / bs * steps)

            result = self._tune(x_train, y_train, x_test, y_test, epochs, trial_steps, lr, bs, weights, loss, metrics)
    
            # get the model and hyperparameters with the best validation accuracy
            # we call this a near-optima point
            val_acc = result[1]
            if val_acc > best[0]:
                best = (val_acc, lr, bs)
                print("\nCurrent Best: lr", lr, "bs", bs)

        # narrow search space to within vicinity of the best near-optima
        learning_rates = [ best[1] / 2, best[1] * 2]
        batch_sizes = [int(best[2] / 2), int(best[2] * 2)]
        for _ in range(trials):
            print("\nNarrowing, Trial", _ + 1)
            lr = learning_rates[random.randint(0, 1)]
            bs = batch_sizes[random.randint(0, 1)]

            # Check for repeat
            if (lr, bs) in tried:
                print("Random Selection already tried", (lr, bs))
                continue
            tried.append( (lr, bs))

            # Adjust steps so each trial sees same number of examples
            trial_steps = int(min_bs / bs * steps)

            result = self._tune(x_train, y_train, x_test, y_test, epochs, trial_steps, lr, bs, weights, loss, metrics)
   
            val_acc = result[1]
            if val_acc > best[0]:
                best = (val_acc, lr, bs)
                print("\nCurrent Best: lr", lr, "bs", bs)

        print("\nSelected Learning Rate", lr, "Batch Size", bs)

        if save is not None:
            with open(save + '/tune/hp.json', 'w') as f:
                data = { 'lr' : lr, 'bs': bs, 'trials': trials }
                json.dump(data, f)
            self.model.save_weights(save + '/tune/chkpt')

        return best[1], best[2]
