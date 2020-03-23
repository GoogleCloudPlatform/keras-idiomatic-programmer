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

class Preprocess:
    ''' Preprocess base (super) class for Composable Models '''

    def __init__(self):
        """ Constructor
        """
        pass

    ###
    # Preprocessing
    ###

    def normalization(self, x_train, x_test=None, centered=False):
        """ Normalize the input
            x_train : training images
            y_train : test images
        """
        if x_train.dtype == np.uint8:
            if centered:
                x_train = ((x_train - 1) / 127.5).astype(np.float32)
                if x_test is not None:
                    x_test  = ((x_test  - 1) / 127.5).astype(np.float32)
            else:
                x_train = (x_train / 255.0).astype(np.float32)
                if x_test is not None:
                    x_test  = (x_test  / 255.0).astype(np.float32)
        return x_train, x_test

    def standardization(self, x_train, x_test=None):
        """ Standardize the input
            x_train : training images
            x_test  : test images
        """
        self.mean = np.mean(x_train)
        self.std  = np.std(x_train)
        x_train = ((x_train - self.mean) / self.std).astype(np.float32)
        if x_test is not None:
            x_test  = ((x_test  - self.mean) / self.std).astype(np.float32)
        return x_train, x_test

    def label_smoothing(self, y_train, n_classes, factor=0.1):
        """ Convert a matrix of one-hot row-vector labels into smoothed versions. 
            y_train  : training labels
            n_classes: number of classes
            factor   : smoothing factor (between 0 and 1)
        """
        if 0 <= factor <= 1:
            # label smoothing ref: https://www.robots.ox.ac.uk/~vgg/rg/papers/reinception.pdf
            y_train *= 1 - factor
            y_train += factor / n_classes
        else:
            raise Exception('Invalid label smoothing factor: ' + str(factor))
        return y_train
