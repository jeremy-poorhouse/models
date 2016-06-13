# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A binary to evaluate Inception on the distracted data set.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

from inception import inception_test
from inception.distracted_data import DistractedData

FLAGS = tf.app.flags.FLAGS


def main(unused_argv=None):
  dataset = DistractedData(subset=FLAGS.subset, examples=int(FLAGS.examples))
  assert dataset.data_files()
  if tf.gfile.Exists(FLAGS.test_dir):
    tf.gfile.DeleteRecursively(FLAGS.test_dir)
  tf.gfile.MakeDirs(FLAGS.test_dir)
  inception_test.test(dataset)


if __name__ == '__main__':
  tf.app.run()
