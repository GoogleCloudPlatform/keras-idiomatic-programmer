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

from tensorflow.keras import Sequential, Input, optimizers
from tensorflow.keras.layers import Dense, Flatten

# Build a DNN multi-class classifier
model = Sequential()
model.add(Flatten(input_shape=(28, 28, 1)))
# … deleted for brevity

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=tf.train.AdamOptimizer(), 
              metrics=['accuracy'])

# Create an estimator model from the compiled model
estimator = tf.keras.estimator.model_to_estimator(model)

# Make an input function for the estimator
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x=x_train,
    y=y_train,
    num_epochs=10,
    shuffle=True)

# Train the estimator version of the model
estimator.train(input_fn=train_input_fn, steps=2000)
