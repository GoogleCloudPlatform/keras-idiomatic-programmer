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

# create an iterator for the on-disk dataset
dataset = tf.data.TFRecordDataset('example.tfrecord')

# create a dictionary description for deserializing a TFRecord
feature_description = {
    'image': tf.FixedLenFeature([], tf.string),
    'label': tf.FixedLenFeature([], tf.int64),
    'shape': tf.FixedLenFeature([], tf.int64),
}

def _parse_function(proto):
    ''' parse the next serialized TFRecord using the feature description '''
    return tf.parse_single_example(proto, feature_description)

parsed_dataset = dataset.map(_parse_function)
