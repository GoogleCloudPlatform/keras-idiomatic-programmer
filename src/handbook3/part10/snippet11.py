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

import random
import numpy as np

# Make a fake dataset of 100 samples of 10 classes (labels)
# x are the images
x = [[_,_] for _ in range(100)]
# y are the corresponding labels, randomly chosen
y = [ random.randint(1,10) for _ in range(100)]

# percent of dataset for training
percent = 0.2 

# find the pivot point in dataset to split
pivot = int(len(x) * (1 - percent))

# presumed to be randomly shuffled
x_train = x[0:pivot]
y_train = y[0:pivot]
x_test  = x[pivot:]
y_test  = y[pivot:]

print("Train", len(x_train), len(y_train)) # will output Train 80 80
print("Test ", len(x_test),  len(y_test))  # will output Test  20 20
