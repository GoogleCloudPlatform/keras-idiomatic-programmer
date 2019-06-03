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

# read in the image
image = Image.open('apple.jpg')

# rotate the image 90 degrees
rotate = image.transpose(Image.ROTATE_90)
rotate.show()

# rotate the image 180 degrees
rotate = image.transpose(Image.ROTATE_180)
rotate.show()

# rotate the image 270 degrees
rotate = image.transpose(Image.ROTATE_270)
rotate.show()
