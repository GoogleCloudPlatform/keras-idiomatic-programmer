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

# The inception branches (where x is the previous layer)
x1 = layers.MaxPooling2D((3, 3), strides=(1,1), padding='same')(x)
x2 = layers.Conv2D(64, (1, 1), strides=(1, 1), padding='same')(x)
x3 = layers.Conv2D(96, (3, 3), strides=(1, 1), padding='same')(x)
x4 = layers.Conv2D(48, (5, 5), strides=(1, 1), padding='same')(x)

# concatenate the filters
x = layers.concatenate([x1, x2, x3, x4])
