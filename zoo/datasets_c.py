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
from training_c import Training

class Dataset(object):
    ''' Dataset base (super) class for Models '''

    def __init__(self):
        """ Constructor
        """
        self.x_train = None
        self.y_train = None
        self.x_test  = None
        self.y_test  = None
        self.n_classes = 0

    @property
    def data(self):
        return (x_train, y_train), (x_test, y_test)

    def load_data(self, train, test=None, std=False, onehot=False, smoothing=0.0):
        """ Load in memory data
            train: expect form: (x_train, y_train)
        """
        self.x_train, self.y_train = train
        if test is not None:
            self.x_test, self.y_test   = test
        if std:
            self.x_train, self.x_test = self.standardization(self.x_train, self.x_test)

        if self.y_train.ndim == 2:
            self.n_classes = np.max(self.y_train) + 1
        else:
            self.n_classes = self.y_train.shape[1]
        if onehot:
            self.y_train = to_categorical(self.y_train, self.n_classes)
            self.y_test  = to_categorical(self.y_test, self.n_classes)
        if smoothing > 0.0:
            self.y_train = self.label_smoothing(self.y_train, self.n_classes, smoothing)

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

