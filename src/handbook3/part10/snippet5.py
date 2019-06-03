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

# open the CSV file and count (using sum) the number of lines, which equals the 
# number of samples
with open(csv_file) as f:
        nimages  = sum(1 for line in f)
        # subtract one from the total count if the first line in CSV file is a header
        if <has header>: 
                nimages -= 1

# create a sequential index between 0 and nimages-1
index = [i for i in range(nimages)]

# now randomly sort the index
random.shuffle(index)