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

from keras.applications import ResNet50
from keras.layers import Dense
from keras import Model

# Get a pre-trained/pre-built model without the classifier and retain the global 
# average pooling # layer following the final convolution (bottleneck) layer
model = ResNet50(include_top=False, pooling='avg', weights='imagenet')

# Freeze the weights of the remaining layer
for layer in model.layers:
    layer.trainable = False

# Add a classifier for 20 classes
output = Dense(20, activation='softmax')(model.output)

# Compile the model for training
model = Model(model.input, output)
model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics=['accuracy'])

# Now train the model
