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


import cv2
import numpy as np
from matplotlib import pyplot as plt

# read in the image
image = cv2.imread('apple.jpg')

# get the height and width of the image
height = image.shape[0]
Width  = image.shape[1]

# shift the image down by 10%
roll = np.roll(image, height // 10, axis=0)
plt.imshow(roll)

# shift the image up by 10%
roll = np.roll(image, -(height // 10), axis=0)
plt.imshow(roll)

# shift the image right by 10%
roll = np.roll(image, width // 10, axis=1)
plt.imshow(roll)

# shift the image left by 10%
roll = np.roll(image, -(width // 10), axis=1)
plt.imshow(roll)

