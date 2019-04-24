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
from matplotlib import pyplot as plt

image = cv2.imread('apple.jpg')

contrast   = 0.5
brightness = 40

# scale the pixel values and then convert back to uint8
image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

plt.imshow(image)
