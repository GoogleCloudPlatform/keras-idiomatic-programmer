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

# Get a pre-built model for input shape (100,100,3) and without the classifier
model = ResNet50(include_top=False, input_shape=(100, 100, 3), pooling='avg')

# Add a classifier for 20 classes
model.add(Dense(20, activation='softmax'))

# Compile the model for training
model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics=['accuracy'])

# Now train the model
