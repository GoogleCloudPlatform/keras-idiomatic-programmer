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


# Let's set aside a copy of the original test data and training data
x_test_copy  = x_test
X_train_copy = x_train

# Let's normalize the training/test data and cast to float32
x_train = (x_train / 255.0).astype(np.float32)
x_test  = (x_test  / 255.0).astype(np.float32)

# Let's reshape for Keras from (H,W) to (H,W,C)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test  = x_test.reshape(-1, 28, 28, 1)

# this will output (60000, 28, 28, 1) and (10000, 28, 28, 1)
print("x_train", x_train.shape, "x_test", x_test.shape)

# We need to convert the labels to one-hot-encoding
y_train = to_categorical(y_train)
y_test  = to_categorical(y_test)

# this will output (60000, 10) and (10000, 10)
print("y_train", y_train.shape, "y_test", y_test.shape)

# This is the inverted test dataset
x_test_invert = np.invert(x_test_copy)
x_test_invert = (x_test_invert / 255.0).astype(np.float32)

# This is the shifted test dataset
x_test_shift = np.roll(x_test_copy, 4)
x_test_shift = (x_test_shift / 255.0).astype(np.float32)

# Let's reshape for Keras from (H,W) to (H,W,C)
x_test_invert = x_test_invert.reshape(-1, 28, 28, 1)
x_test_shift  = x_test_shift.reshape(-1, 28, 28, 1)

def DNN(nodes, dropout=False):
  model = Sequential()
  model.add(Flatten(input_shape=(28, 28, 1)))
  for n_nodes in nodes:
    model.add(Dense(n_nodes))
    model.add(ReLU())
    if dropout:
      model.add(Dropout(0.5))
      dropout /= 2.0
  model.add(Dense(10))
  model.add(Activation('softmax'))

  # For a multi-class classification problem
  model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
  model.summary()
  return model

model = DNN([512])
model.fit(x_train, y_train, epochs=10, batch_size=32, shuffle=True, verbose=2)
score = model.evaluate(x_test, y_test, verbose=1)
print("test", score)

score = model.evaluate(x_test_invert, y_test, verbose=1)
print("inverted", score)
score = model.evaluate(x_test_shift, y_test, verbose=1)
print("shifted", score)

model = DNN([1024])
model.fit(x_train, y_train, epochs=10, batch_size=32, shuffle=True, verbose=2)
score = model.evaluate(x_test, y_test, verbose=1)
print("test", score)

score = model.evaluate(x_test_invert, y_test, verbose=1)
print("inverted", score)
score = model.evaluate(x_test_shift, y_test, verbose=1)
print("shifted", score)

model = DNN([512, 512], True)
model.fit(x_train, y_train, epochs=10, batch_size=32, shuffle=True, verbose=2)
score = model.evaluate(x_test, y_test, verbose=1)
print("test", score)

score = model.evaluate(x_test_invert, y_test, verbose=1)
print("inverted", score)
score = model.evaluate(x_test_shift, y_test, verbose=1)
print("shifted", score)

