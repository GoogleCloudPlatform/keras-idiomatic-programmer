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
import os

def loadImages(subdir, channels, shape):
        images = []

        # get list of all files in subdirectory cats
        files = os.scandir(subdir)
        # read each image in and append the in-memory image to a list
        for file in files:
                   # convert to grayscale
                   if channels == 1:
                        image = cv2.imread(file.path, cv2.IMREAD_GRAYSCALE)
                   # convert to RGB color
                   else:
                        image = cv2.imread(file.path, cv2.IMREAD_COLOR)
                   # resize the image to the target input shape
                   images.append(cv2.resize(image, shape, cv2.INTER_AREA))
        return images

loadImages('cats', 3, (128, 128))
