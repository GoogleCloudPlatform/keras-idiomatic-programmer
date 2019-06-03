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

# create a checkpoint for each epoch
checkpoint = tf.keras.callbacks.ModelCheckpoint('weights.{epoch:02}')

model.fit(dataset, epochs=10, steps_per_epoch=len(x_train)//32,  
          validation_data=val_dataset, validation_steps=len(x_test)//32, 
          callbacks=[checkpoint])