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
# assume the dataset appears as [x, x, x, x] and [y, y, y, y], where x is the image 
# data and y is the corresponding label

# pick an arbitrary number for a seed
seed = 101

# seed the random sequence and shuffle the x (image) data
np.random.seed(seed)
np.random.shuffle(x_data)

# reset the same seed to get the identical random sequence and shuffle the y
# (label) data
np.random.seed(seed)
np.random.shuffle(y_data)
