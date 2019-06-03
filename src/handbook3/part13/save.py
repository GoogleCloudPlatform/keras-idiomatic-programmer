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
# imports as tf.keras
import tensorflow as tf
from tensorflow.keras import Sequential, Input, optimizers
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np

# Build a DNN multi-class classifier
model = Sequential()
model.add(Flatten(input_shape=(28, 28, 1)))
# lines deleted for brevity
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=tf.train.AdamOptimizer(),  
              metrics=['accuracy'])

# Train the model
model.fit(dataset, epochs=10, steps_per_epoch=len(x_train)//32,  
          validation_data=val_dataset, validation_steps=len(x_test)//32)

# save the model weights in TF checkpoint format.
model.save_weights('my-model-weights')

# later restore the weights
mode.load_weights('my-model-weights')
