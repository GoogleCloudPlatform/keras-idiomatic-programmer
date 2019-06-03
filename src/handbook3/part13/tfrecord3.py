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

def TFRecordImageNormalized(path, label):
        ''' The normalized version of the image '''
    
        # read in (uncompress) the image and normalize the pixel data
        image = (cv2.imread(path) / 255.0).astype(np.float32)
        shape = image.shape
         
        # make the record
        return tf.train.Example(features = tf.train.Features(feature = {
        'image': tf.train.Feature(bytes_list = tf.train.BytesList(value = 
                                  [image.tostring()])),
        'label': tf.train.Feature(int64_list = tf.train.Int64List(value = [label])),
        'shape': tf.train.Feature(int64_list = tf.train.Int64List(value = 
                                  [shape[0], shape[1], shape[2]]))
        }))

example = TFRecordImageNormalized('example.jpg', 0)
# output should be something like: 2,000,000
print(example.ByteSize())
