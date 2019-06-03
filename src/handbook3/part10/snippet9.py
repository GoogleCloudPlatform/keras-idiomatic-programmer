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

from keras.preprocessing.image import ImageDataGenerator

# instantiate an Image Data generator object and specify standardizing the image
# data
datagen = ImageDataGenerator(featurewise_std_normalization=True)

# calculate the mean/stddev for standardization
datagen.fit(x_train)

# feed (train) the neural network
