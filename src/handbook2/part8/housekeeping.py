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

# Keras classes we will use to build models
from keras import Sequential, Input
from keras.layers import Flatten, Dense, Activation, ReLU, Conv2D, MaxPooling2D, Dropout
from keras.utils import to_categorical

# Keras library for the builtin MNIST dataset
from keras.datasets import mnist

# Other python modules for preparation/manipulating the images
import numpy as np
import random
import cv2

# Get the builtin dataset from Keras
(x_train, y_train), (x_test, y_test) = mnist.load_data()
