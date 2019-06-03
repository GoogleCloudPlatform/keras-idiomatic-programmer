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

# Create a writer for writing a TFRecord in protocol buffer format
with tf.python_io.TFRecordWriter('example.tfrecord') as writer:
        writer.write(example.SerializeToString())
        
with tf.python_io.TFRecordWriter('example.tfrecord') as writer:
        # examples is a list of TFRecords, one per image
        for example in examples:
            writer.write(example.SerializeToString())
           
# create an iterator for iterating through TFRecords in sequential order
iterator = tf.python_io.tf_record_iterator('example.tfrecord')
for record in iterator:
        # each record is read in as a serialized string
        example = tf.train.Example()
        # convert the serialized string to a TFRecord
        example.ParseFromString(record)
        

