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

from PIL import Image
image = Image.open('apple.jpg')

zoom = 2 # zoom by factor of 2
# remember the original height, width of the image
height, width = image.size

# resize (scale) the image proportional to the zoom
image = image.resize( (int(height*zoom), int(width*zoom)), Image.BICUBIC)

# find the center of the scaled image
center = (image.size[0]//2, image.size[1]//2)

# calculate the crop upper left corner
crop = (int(center[0]//zoom), int(center[1]//zoom))

# calculate the crop bounding box
box = ( crop[0], crop[1], (center[0] + crop[0]), (center[1] + crop[1]) )

image = image.crop( box )
image.show()
