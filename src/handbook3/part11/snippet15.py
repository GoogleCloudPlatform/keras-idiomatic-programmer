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

from keras.datasets import cifar100
import numpy as np

# get the train and test data
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='coarse')

# normalize the pixel data between 0 and 1
x_train = (x_train / 255.0).astype(np.float32)
x_test  = (x_test / 255.0).astype(np.float32)

# convert the labels to categorical
nclasses = np.max(y_train) + 1
y_train = utils.to_categorical(y_train, nclasses)
y_test  = utils.to_categorical(y_test, nclasses)

# further split off from the training data the validation data (10%)
pivot = int(len(x_train) * 0.9)
x_val = x_train[pivot:]
y_val = y_train[pivot:]
x_train = x_train[:pivot]
y_train = y_train[:pivot]
