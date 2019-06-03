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

# model is a compiled keras model (Model or Sequential class).
model.fit(x_train, y_train, batch_size=32)

# randomly rotate images +/- 20 degrees
datagen = ImageDataGenerator(rotation_range=20)

# train the model
model.fit_generator(datagen.flow(x_train, y_train, batch_size=32), epochs=10)