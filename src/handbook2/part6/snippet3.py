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
import os

def loadDirectory(parent):
        classes = {}  # list of class to label mappings
        dataset = []  # list of images by class

        # get list of all subdirectories under the parent (root) directory of the
        # dataset
        for subdir in os.scandir(parent):
                # ignore any entry that is not a subdirectory (e.g., license file)
                if not subdir.is_dir():
                        continue

                # maintain mapping of class (subdirectory name) to label (index)
                classes[subdir.name] = len(dataset)

                # maintain list of images by class
                dataset.append(loadImages(subdir.path))

                print("Processed:", subdir.name, "# Images",
                      len(dataset[len(dataset)-1]))

        return dataset, classes

loadDirectory('cats_n_dogs')

