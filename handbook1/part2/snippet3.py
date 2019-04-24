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

from keras import Input, Model
from keras.layers import Dense
from keras.layers import Conv2D, MaxPooling2D, Flatten

# Create the input vector (128 x 128).
inputs = Input(shape=(128, 128, 1))
layer  = Conv2D(16, kernel_size=(3, 3), strides=(2, 2), padding="same",
                activation='relu')(inputs)
layer  = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(layer)
layer  = Flatten()(layer)
layer  = Dense(512, activation='relu')(layer)
output = Dense(26, activation='softmax')(layer)

# Now let's create the neural network, specifying the input layer and output layer.
model = Model(inputs, output)
