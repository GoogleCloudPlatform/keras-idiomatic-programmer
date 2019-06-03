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
import tensorflow as tf
import numpy as np
import sys
import cv2

def TFRecordImage(path, label):
        ''' The original compressed version of the image '''
    
        # read in (and uncompress) the image to get its shape
        image = cv2.imread(path)
        shape = image.shape
    
        # read in the image a second time for the original bytes (not uncompressed)
        with tf.gfile.FastGFile(path, 'rb') as fid:
            disk_image = fid.read()
         
        # make the record
        return tf.train.Example(features = tf.train.Features(feature = {
        'image': tf.train.Feature(bytes_list = tf.train.BytesList(value = 
                                  [disk_image])),
        'label': tf.train.Feature(int64_list = tf.train.Int64List(value = [label])),
        'shape': tf.train.Feature(int64_list = tf.train.Int64List(value = 
                                  [shape[0], shape[1], shape[2]]))
        }))

example = TFRecordImage('example.jpg', 0)
# output would be something like: 25,000
print(example.ByteSize())
