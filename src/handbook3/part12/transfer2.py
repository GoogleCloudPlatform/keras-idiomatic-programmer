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
import keras.layers

# Get a pre-trained/pre-built model without the classifier and retain the global 
# average pooling # layer following the final convolution (bottleneck) layer
model = ResNet50(include_top=False, pooling='avg', weights='imagenet')

# Freeze the weights of the remaining layer
for layer in model.layers:
    Layer.trainable = False

# Add a classifier for 20 classes
output = Dense(20, activation='softmax')(model.output)

# Compile the model for training
model = Model(model.input, output)
model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics=['accuracy'])

# Train the classifier
model.fit(x_data, y_data, batch_size=32, epochs=50, validation_split=0.2)

stem = None # layer that is the convolutional layer for the stem group
groups = [] # the add layer for each convolutional group
conv2d = [] # the convolutional layers of a group

first_conv2d = True
for layer in model.layers:
        if type(layer) == layers.convolutional.Conv2D:
        # In ResNet50, the first Conv2D is the stem convolutional layer
        if first_conv2d == True:
                stem = layer
                first_conv2d = False
        # Keep list of convolutional layers per convolutional group
        else:
                conv2d.append(layer)
        # Each convolutional group in Residual Networks ends with a Add layer.
        # Maintain list in reverse order (top-most conv group is top of list)
        elif type(layer) == layers.merge.Add:
                groups.insert(0, conv2d)
                conv2d = []

# Unfreeze a convolutional group at a time (from top-most to bottom-most)
# And fine-tune (train) that layer
for i in range(1, len(groups)):
       # Unfreeze the convolutional layers in this conv/residual group
        for layer in groups[i]:
            layer.trainable = True

       # re-compile the model for training
       model.compile(loss='categorical_crossentropy', optimizer='adam', 
                     metrics=['accuracy'])
    
        # Fine-tune train the convolutional group(s)
        model.fit(x_data, y_data, batch_size=32, epochs=5)

# Unfreeze the stem convolutional and do a final fine-tuning
stem.trainable = True
model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics=['accuracy'])
model.fit(x_data, y_data, batch_size=32, epochs=5, validation_split=0.2)
