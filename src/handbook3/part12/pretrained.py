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
from keras.applications.resnet import preprocess_input, decode_predictions

# Get a pre-built ResNet50 model
model = ResNet50(weights='imagenet')

# Read the image into memory as a numpy array
image = cv2.imread('elephant.jpg', cv2.IMREAD_COLOR)

# Resize the image to fit the input shape of ResNet model
image = cv2.resize(image, (224, 224), cv2.INTER_LINEAR)

# Preprocess the image using the same image processing used by the pre-built model
image = preprocess_input(image)

# Reshape from (224, 224, 3) to (1, 224, 224, 3) for the predict() method
image = image.reshape((-1, 224, 224, 3))

# Call the predict() method to classify the image
predictions = model.predict(image)

# Display the class name based on the predicted label using the decode function for
# the built-in model.
print(decode_predictions(preds, top=3))
