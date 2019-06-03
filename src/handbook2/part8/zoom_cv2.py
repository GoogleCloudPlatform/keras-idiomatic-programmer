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

zoom = 2 # zoom by factor of 2
image = cv2.imread('apple.jpg')

# remember the original height, width of the image
height, width = image.shape[:2]

# find the center of the scaled image
center = (image.shape[0]//2, image.shape[1]//2)
z_height = int(height // zoom)
z_width  = int(width  // zoom)

# slice (cutout) the zoomed image by forming a crop bounding box
image = image[(center[0] - z_height//2):(center[0] + z_height//2), center[1] -
              z_width//2:(center[1] + z_width//2)]

# resize (enlarge) the cropped image back to the original size.
image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

plt.imshow(image)

