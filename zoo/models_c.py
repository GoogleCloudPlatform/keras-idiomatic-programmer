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
from datasets_c import Dataset

class Composable(Layers, Preprocess, Pretraining, HyperTune, Training, Dataset):
    ''' Composable base (super) class for Models '''

    def __init__(self, input_shape, include_task, default_hyperparameters=None, **hyperparameters):
        """ Constructor
            input_shape  : input tensor to the model
            include_task : include the task component
            default_hyperparameters: parent model default hyperparameter settings
            **hyperparameters: overridden hyperparameter settings
        """
        for key, value in hyperparameters.items():
            default_hyperparameters[key] = value
        Layers.__init__(self, **default_hyperparameters)
        Preprocess.__init__(self)
        Pretraining.__init__(self)
        HyperTune.__init__(self)
        Training.__init__(self)
        Dataset.__init__(self)

        self.input_shape = input_shape
        self.include_task = include_task

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

    @property
    def softmax(self):
        return self._softmax

    @softmax.setter
    def softmax(self, layer):
        self._softmax = layer

   
    @property
    def inputs(self):
        return self._model.inputs


