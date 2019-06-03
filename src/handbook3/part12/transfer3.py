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
from keras import Model
from keras.layers import Dense, Flatten
from keras.models import load_model, model_from_json

model = ResNet50(include_top=False, pooling=None, input_shape=(100, 100, 3))

# save the base model
base_model = model.to_json()# Write the JSON string to a file
with open('produce-model.json', 'w') as f:  
        f.write(base_model)

# Add classifier
output = Flatten(name='bottleneck')(model.output)
output = Dense(20, activation='softmax')(output)

# do training here

# save the model weights
model.save_weights('produce-weights.h5')

# Read the JSON string for the base model from a file
with open('produce-model.json', 'r') as f:  
        base_model = f.read()

# Reuse the base model and trained weights
model = model_from_json(base_model)
model.load_weights('produce-weights.h5')

# Add classifier
output = Flatten(name='bottleneck')(model.output)
output = Dense(20, activation='softmax')(output)

# Compile the model
model = Model(model.input, output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
# train the new model for a new dataset
