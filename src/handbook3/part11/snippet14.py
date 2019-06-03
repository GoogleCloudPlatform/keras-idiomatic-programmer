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


from keras.models import load_weights
from keras.models import model_from_json

# Read the JSON string from a file
with open('mymodel.json', 'r') as f:  
         s = f.read()

# Load the model architecture
model = model_from_json(s)

# Load the trained weights for the model
model.load_weights('mymodel-weights.h5')
