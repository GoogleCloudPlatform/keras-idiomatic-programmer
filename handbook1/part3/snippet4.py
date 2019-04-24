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


# Inception v3 stem, 7 x 7 is replaced by a stack of 3 x 3 convolutions.
x = layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same')(input)
x = layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same')(x)
x = layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)

# max pooling layer
x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

# bottleneck convolution
x = layers.Conv2D(80, (1, 1), strides=(1, 1), padding='same')(x)

# next convolution layer
x = layers.Conv2D(192, (3, 3), strides=(1, 1), padding='same')(x)

# strided convolution - feature map reduction
x = layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same')(x)

