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
from keras import layers, Input

# not the input shape expected by the stem (which is (230, 230, 3)
inputs = Input(shape=(224, 224, 3))

# pre-stem
inputs = layers.ZeroPadding2D((3, 3))(inputs)
# this will output: (230, 230, 3)
print(inputs.shape)

# this stem's expected shape is (230, 230, 3)
x = layers.Conv2D(64, (7, 7), strides=(2,2))(inputs)
X = layers.BatchNormalization()(x)
x = layers.ReLU()(x)

# this will output: (?, 112, 112, 64)
print(x.shape)
