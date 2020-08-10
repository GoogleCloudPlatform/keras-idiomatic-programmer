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

    def init_draw(self, x_train=None, y_train=None, ndraws=5, epochs=3, steps=350, lr=1e-06, 
                  batch_size=32, metric='loss', early=False, save=None):
        """ Use the lottery ticket principle to find the best weight initialization
            x_train : training images
            y_train : training labels
            ndraws  : number of draws to find the winning lottery ticket
            epochs  : number of trial epochs
            steps   : number of steps per epoch
            lr      : tiny learning rate
            batch_size: batch size
            metric  : metric used for determining best draw
            early   : whether to early stop when best draw found
            save    : file to save initialized weights to
        """
        print("\n*** Initialize Draw")

        if x_train is None:
            x_train = self.x_train
            y_train = self.y_train

        loss = sys.float_info.max
        acc  = 0
        w_best = None

        # previous values
        prev = None
        p_draws = 0
        if save is not None:
            for path in [ save, save + '/init']:
                try:
                    os.mkdir(path)
                except:
                    pass
            if os.path.exists(save + '/init/best.json'):
                with open(save + '/init/best.json', 'r') as f:
                    data = json.load(f)
                    loss = float(data['loss'])
                    acc  = float(data['acc'])
                    p_draws = int(data['ndraws'])
                    self.model.load_weights(save + '/init/chkpt')
                    w_best = self.model.get_weights()
                    print("Previous best, loss =", loss, 'acc = ', acc)

                    try:
                        prev = [ data['prev'], { 'loss': loss, 'acc': acc, 'ndraws': p_draws } ]
                    except:
                        prev = { 'loss': loss, 'acc': acc, 'ndraws': p_draws }

        for _ in range(ndraws):
            self.model = tf.keras.models.clone_model(self.model)
            self.compile(optimizer=Adam(lr))
            w = self.model.get_weights()

            # Create generator for training in steps
            datagen = ImageDataGenerator()

            print("\n*** Lottery", _ + 1, "of", ndraws)
            self.model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
                                        epochs=epochs, steps_per_epoch=steps, verbose=1)

            # Next Best
            d_loss = self.model.history.history['loss'][epochs-1]
            d_acc  = self.model.history.history['acc'][epochs-1]
            if d_loss < loss:
                loss = d_loss
                acc  = d_acc
                w_best = self.model.get_weights()
                print("\n*** Current Best:", metric, loss) 
                if early:
                    ndraws = _ + 1
                    break
                if save is not None:
                    self._save_best(save, loss, acc, p_draws + _ + 1, epochs, steps, prev)

        # Set the best
        if w_best is not None:
            self.model.set_weights(w_best)
 
            # Save the initialized weights
            if save is not None:
                self._save_best(save, loss, acc, p_draws + ndraws, epochs, steps, prev)
        print("\n*** Selected Draw:", metric, loss) 

    def _save_best(self, save, loss, acc, ndraws, epochs, steps, prev=None):
        """ Save current best weights
            save  : directort to save weights
            loss  : metric information
            acc   : metric information
            ndraws: total number of draws
            epochs: number of epochs
            steps : number of steps per epoch
            prev  : previous results
        """
        # Late Resetting
        self.model.save_weights(save + '/init/chkpt')
        with open(save + "/init/best.json", "w") as f:
            if prev is None:
                data = {'loss': loss, 'acc': acc, 'ndraws': ndraws, 'epochs': epochs, 'steps': steps}
            else:
                data = {'loss': loss, 'acc': acc, 'ndraws': ndraws, 'epochs': epochs, 'steps': steps, 'prev': prev}
            data = json.dumps(data)
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
            save      : file to save warmup weights 
        """
        print("\n*** Warmup (for numerical stability)")

        if x_train is None:
            x_train = self.x_train
            y_train = self.y_train

        # Load selected weight initialization draw
        if save is not None:
            for path in [ save, save + '/warmup']:
                try:
                    os.mkdir(path)
                except:
                    pass
            if os.path.exists(save + '/init/chkpt.index'):
                self.model.load_weights(save + '/init/chkpt')
                print("Load weights from Lottery Draw initialization")

        # Setup learning rate scheduler
        self.compile(optimizer=Adam(s_lr), loss=loss, metrics=metrics)
        lrate = LearningRateScheduler(self.warmup_scheduler, verbose=1)
        self.w_epochs = epochs
        self.w_lr     = e_lr - s_lr

        # Train the model
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1,
                       callbacks=[lrate])

        if save is not None:
            self.model.save_weights(save + '/warmup/chkpt')
            with open(save + '/warmup/hp.json', 'w') as f:
                data = {'s_lr': s_lr, 'e_lr': e_lr, 'epochs': epochs }
                json.dump(data, f)

    def pretext(self, x_train= None, zigsaw=9, epochs=10, batch_size=32, lr=0.001, 
                loss='mse', metrics=['mse'], save=None):
        """ Pretrain using unsupervised pre-text task for zigsaw puzzle to learn essential features
            x_train   : training images
            zigsaw    : number of tiles in zigsaw puzzle
            epochs    : number of epochs for pretext task training
            batch_size: batch size
            lr        : pre-text learning rate
            loss      : loss function
            metrics   : training metrics to report
            save      : file to save pretext weights            
        """
        print("\n*** Pretext Task (for essential features)")

        if x_train is None:
            x_train = self.x_train

        # Load selected weight after hypertune
        if save is not None:
            for path in [ save, save + '/pretext']:
                try:
                    os.mkdir(path)
                except:
                    pass
            if os.path.exists(save + '/tune/chkpt.index'):
                self.model.load_weights(save + '/tune/chkpt')
            elif os.path.exists(save + '/warmup/chkpt.index'):
                self.model.load_weights(save + '/warmup/chkpt')
            elif os.path.exists(save + '/init/chkpt.index'):
                self.model.load_weights(save + '/init/chkpt')

            if lr is None:
                with open(save + '/tune/hp.json') as f:
                   data = json.load(f)
                   lr = data['lr']
                   batch_size = data['bs']

        # Get the pooling layer before the final dense output layer
        pooling = self.model.layers[len(self.model.layers)-2].output

        # Attach a new top for the zigsaw puzzle
        outputs = self.Dense(pooling, zigsaw)
        self.relu = zigsaw
        outputs = self.ReLU(outputs)

        # Construct wrapper model with the new top layer
        wrapper = Model(self.model.inputs, outputs)
        wrapper.compile(loss=loss, optimizer=Adam(lr=lr), metrics=metrics)

        # Rows/Columns
        R = x_train.shape[1]
        C = x_train.shape[2]

        # Slicing
        if zigsaw == 4:
            M = int(x_train.shape[1] / 2)
            N = int(x_train.shape[2] / 2)
            ix = [0, 1, 2, 3]
        elif zigsaw == 9:
            M = int(x_train.shape[1] / 3)
            N = int(x_train.shape[2] / 3)
            ix = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        px_train = []
        py_train = []
        for _ in range(len(x_train)):
            tiles = [x_train[_][x:x+M,y:y+N] for x in range(0,R,M) for y in range(0,C,N)]
            random.shuffle(ix)
            if zigsaw == 4:
                r1 = np.concatenate((tiles[ix[0]], tiles[ix[1]]))
                r2 = np.concatenate((tiles[ix[2]], tiles[ix[3]]))
                image = np.concatenate((r1, r2), axis=1)
            else:
                r1 = np.concatenate((tiles[ix[0]], tiles[ix[1]], tiles[ix[2]]))
                r2 = np.concatenate((tiles[ix[3]], tiles[ix[4]], tiles[ix[5]]))
                r3 = np.concatenate((tiles[ix[6]], tiles[ix[7]], tiles[ix[8]]))
                image = np.concatenate((r1, r2, r3), axis=1)
            px_train.append(image)
            py_train.append(ix)

        px_train = np.asarray(px_train)
        py_train = np.asarray(py_train)

        # Train the model
        wrapper.fit(px_train, py_train, epochs=epochs, batch_size=batch_size, verbose=1)

        if save is not None:
            self.model.save_weights(save + '/pretext/chkpt')
