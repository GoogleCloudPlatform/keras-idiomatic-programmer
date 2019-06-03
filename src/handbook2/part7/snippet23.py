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

# Create generator for feeding neural network from on-disk, and
# resize images to 128 x 128 as being feed
feeder = datagen.flow_from_directory('root_of_dataset', target_size=(128, 128))

# Train the neural network by calling the fit_generator() method with the above feeder
model.fit_generator(feeder, steps_per_epoch=steps, epochs=epochs)
