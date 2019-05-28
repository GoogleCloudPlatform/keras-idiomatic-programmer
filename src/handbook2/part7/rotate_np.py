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


import numpy as np
import cv2
from matplotlib import pyplot as plt

# read in the image
image = cv2.imread('apple.jpg')

# rotate the image 90 degrees
rotate = np.rot90(image, 1)
plt.imshow(rotate)

# rotate the image 180 degrees
rotate = np.rot90(image, 2)
plt.imshow(rotate)

# rotate the image 270 degrees
rotate = np.rot90(image, 3)
plt.imshow(rotate)

