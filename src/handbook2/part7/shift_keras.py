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
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Let's make a batch of 1 image (apple)
image = cv2.imread('apple.jpg')
batch = np.asarray([image])

# Create a data generator for augmenting the data
datagen = ImageDataGenerator(width_shift_range=0.2, height_shift_range=0.2)

# Let's run the generator, where every image is a random augmentation
step=0
for x in datagen.flow(batch, batch_size=1):
        step += 1
        if step > 6: break
        plt.figure()
        # the augmentation operation will change the pixel data to float
        # change it back to uint8 for displaying the image
        plt.imshow(x[0].astype(np.uint8))

