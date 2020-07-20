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

class Pretraining(object):
    ''' Pretraining base (super) class for Composable Models '''

    def __init__(self):
        """ Constructor
        """
        pass

    ###
    # Pre-Training
    ###

    # training variables
    w_lr           = 0    # target warmup rate
    w_epochs       = 0    # number of epochs in warmup

    def init_draw(self, x_train=None, y_train=None, ndraws=5, epochs=3, steps=350, lr=1e-06, batch_size=32, 
                  metric='loss', save=None):
        """ Use the lottery ticket principle to find the best weight initialization
            x_train : training images
            y_train : training labels
            ndraws  : number of draws to find the winning lottery ticket
            epochs  : number of trial epochs
            steps   : number of steps per epoch
            lr      : tiny learning rate
            batch_size: batch size
            metric  : metric used for determining best draw
            save    : file to save initialized weights to
        """
        if x_train is None:
            x_train = self.x_train
            y_train = self.y_train

        loss = sys.float_info.max
        w_best = None
        if save is not None:
            try:
                os.mkdir(save)
            except:
                pass
            if os.path.exists(save + '/best.json'):
                with open(save + '/best.json', 'r') as f:
                    data = json.load(f)
                    loss = float(data['loss'])
                    self.model.load_weights(save + '/chkpt')
                    w_best = self.model.get_weights()

        print("*** Initialize Draw")
        for _ in range(ndraws):
            self.model = tf.keras.models.clone_model(self.model)
            self.compile(optimizer=Adam(lr))
            w = self.model.get_weights()

            # Create generator for training in steps
            datagen = ImageDataGenerator()

            print("\n*** Lottery", _ + 1)
            self.model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
                                                  epochs=epochs, steps_per_epoch=steps, verbose=1)

            # Next Best
            d_loss = self.model.history.history[metric][epochs-1]
            if d_loss < loss:
                loss = d_loss
                w_best = self.model.get_weights()
                print("\n*** Current Best:", metric, loss) 
                if save is not None:
                    self._save_best(save, loss)

        # Set the best
        if w_best is not None:
            self.model.set_weights(w_best)
 
            # Save the initialized weights
            if save is not None:
                self._save_best(save, loss)
        print("\n*** Selected Draw:", metric, loss) 

    def _save_best(self, save, best):
        """ Save current best weights
            save : directort to save weights
            best : metric information
        """
        self.model.save_weights(save + '/chkpt')
        best = {'loss': best}
        with open(save + "/best.json", "w") as f:
            data = json.dumps(best)
            f.write(data)

    def warmup_scheduler(self, epoch, lr):
        """ learning rate schedular for warmup training
            epoch : current epoch iteration
            lr    : current learning rate
        """
        if epoch == 0:
           return lr
        if epoch == 2:
            # loss is diverging
            if self.model.history.history['loss'][1] > self.model.history.history['loss'][0]:
                print("*** Loss is diverging, Reducing Warmnup Rate")
                self.w_lr /= 10
        return epoch * self.w_lr / self.w_epochs

    def warmup(self, x_train=None, y_train=None, epochs=5, batch_size=32, s_lr=1e-6, e_lr=0.001, 
               loss='categorical_crossentropy', metrics=['acc'], save=None):
        """ Warmup for numerical stability
            x_train   : training images
            y_train   : training labels
            epochs    : number of epochs for warmup
            batch_size: batch size
            s_lr      : start warmup learning rate
            e_lr      : end warmup learning rate
            loss      : loss function
            metrics   : training metrics to report
        """
        if x_train is None:
            x_train = self.x_train
            y_train = self.y_train

        # Load selected weight initialization draw
        if save is not None:
            try:
                os.mkdir(save)
            except:
                pass
            self.model.load_weights(save + '/chkpt')

        print("*** Warmup (for numerical stability)")
        # Setup learning rate scheduler
        self.compile(optimizer=Adam(s_lr), loss=loss, metrics=metrics)
        lrate = LearningRateScheduler(self.warmup_scheduler, verbose=1)
        self.w_epochs = epochs
        self.w_lr     = e_lr - s_lr

        # Train the model
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1,
                       callbacks=[lrate])

        if save is not None:
            self.model.save_weights(save + '/chkpt')

