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
from sklearn.model_selection import train_test_split

# Make a fake dataset of 100 samples of 10 classes (labels)
# x are the images
x = [[_,_] for _ in range(100)]
# y are the corresponding labels, randomly chosen
y = [ random.randint(1,10) for _ in range(100)]

# Split the dataset (x, y) into 80% training and 20% test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, 
                                                    shuffle=True)

print("Train", len(x_train), len(y_train)) # will output Train 80 80
print("Test ", len(x_test),  len(y_test))  # will output Test  20 20
