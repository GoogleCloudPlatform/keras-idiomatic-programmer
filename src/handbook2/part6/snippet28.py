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

# Shuffle the Data (Image Data and Labels)
indices = [_ for _ in range(len(dataset))]
random.shuffle(indices)

# Let's reassemble of list of images and corresponding labels according to our shuffle order in 'indices'
X =[]
Y =[]
for _ in indices:
    X.append(dataset[_])
    Y.append(labels[_])
    
X = np.asarray(X)
Y = np.asarray(Y)
