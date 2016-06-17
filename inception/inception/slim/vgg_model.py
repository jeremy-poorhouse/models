# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""VGG-16 expressed in TensorFlow-Slim.

  Usage:

  # Parameters for BatchNorm.
  batch_norm_params = {
      # Decay for the batch_norm moving averages.
      'decay': BATCHNORM_MOVING_AVERAGE_DECAY,
      # epsilon to prevent 0s in variance.
      'epsilon': 0.001,
  }
  # Set weight_decay for weights in Conv and FC layers.
  with slim.arg_scope([slim.ops.conv2d, slim.ops.fc], weight_decay=0.00004):
    with slim.arg_scope([slim.ops.conv2d],
                        stddev=0.1,
                        activation=tf.nn.relu,
                        batch_norm_params=batch_norm_params):
      # Force all Variables to reside on the CPU.
      with slim.arg_scope([slim.variables.variable], device='/cpu:0'):
        logits, endpoints = slim.inception.inception_v3(
            images,
            dropout_keep_prob=0.8,
            num_classes=num_classes,
            is_training=for_training,
            restore_logits=restore_logits,
            scope=scope)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from inception.slim import ops
from inception.slim import scopes


def vgg_16(inputs,
                 dropout_keep_prob=0.8,
                 num_classes=10,
                 is_training=True,
                 restore_logits=True,
                 scope=''):
  """Latest Inception from http://arxiv.org/abs/1512.00567.

    "Rethinking the Inception Architecture for Computer Vision"

    Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens,
    Zbigniew Wojna

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    dropout_keep_prob: dropout keep_prob.
    num_classes: number of predicted classes.
    is_training: whether is training or not.
    restore_logits: whether or not the logits layers should be restored.
      Useful for fine-tuning a model with different num_classes.
    scope: Optional scope for op_scope.

  Returns:
    a list containing 'logits', 'aux_logits' Tensors.
  """
  # end_points will collect relevant activations for external use, for example
  # summaries or losses.
  dropout_keep_prob = 0.4 if training else 1.0
  
  end_points = {}
  with tf.op_scope([inputs], scope, 'vgg_16'):
    with scopes.arg_scope([ops.conv2d, ops.fc, ops.batch_norm, ops.dropout],
                          is_training=is_training):
      with scopes.arg_scope([ops.conv2d, ops.max_pool, ops.avg_pool],
                            stride=1, padding='SAME'):
        # assume input_op shape is 224x224x3
        # block 1 -- outputs 112x112x64
        end_points['conv1_1'] = ops.conv2d(inputs, 64, [3, 3], stride=1,
                                         scope='conv1_1')
        end_points['conv1_2'] = ops.conv2d(end_points['conv1_1'], 64, [3, 3],
                                         scope='conv1_2')
        end_points['pool1'] = ops.max_pool(end_points['conv1_2'], [2, 2],
                                          stride=2, scope='pool1')

        # block 2 -- outputs 56x56x128
        end_points['conv2_1'] = ops.conv2d(end_points['pool1'], 128, [3, 3],
                                         scope='conv2_1')
        end_points['conv2_2'] = ops.conv2d(end_points['conv2_1'], 128, [3, 3],
                                         scope='conv2_2')
        end_points['pool2'] = ops.max_pool(end_points['conv2_2'], [2, 2],
                                           stride=2, scope='pool2')
        # block 3 -- outputs 28x28x256
        end_points['conv3_1'] = ops.conv2d(end_points['pool2'], 256, [3, 3],
                                         scope='conv3_1')
        end_points['conv3_2'] = ops.conv2d(end_points['conv3_1'], 256, [3, 3],
                                         scope='conv3_2')
        end_points['pool3'] = ops.max_pool(end_points['conv3_2'], [2, 2],
                                           stride=2, scope='pool3')

        # block 4 -- outputs 14x14x512
        end_points['conv4_1'] = ops.conv2d(end_points['pool3'], 512, [3, 3],
                                         scope='conv4_1')
        end_points['conv4_2'] = ops.conv2d(end_points['conv4_1'], 512, [3, 3],
                                         scope='conv4_2')
        end_points['pool4'] = ops.max_pool(end_points['conv4_2'], [2, 2],
                                           stride=2, scope='pool4')

        # block 5 -- outputs 7x7x512
        end_points['conv5_1'] = ops.conv2d(end_points['pool4'], 512, [3, 3],
                                         scope='conv5_1')
        end_points['conv5_2'] = ops.conv2d(end_points['conv5_1'], 512, [3, 3],
                                         scope='conv5_2')
        end_points['pool5'] = ops.max_pool(end_points['conv5_2'], [2, 2],
                                           stride=2, scope='pool5')

        net = end_points['pool5']

        # Final pooling and prediction
        with tf.variable_scope('logits'):
          # flatten
          net = ops.flatten(net, scope='flatten')

          # fully connected
          end_points['fc6'] = ops.fc(net, 1000, activation=None, scope='fc6',
                          restore=restore_logits)
          end_points['fc6_drop'] = ops.dropout(end_points['fc6'], dropout_keep_prob, scope='fc6_drop')

          end_points['fc7'] = ops.fc(end_points['fc6_drop'], 50, activation=None, scope='fc7',
                          restore=restore_logits)
          end_points['fc7_drop'] = ops.dropout(end_points['fc7'], dropout_keep_prob, scope='fc7_drop')

          end_points['fc8'] = ops.fc(end_points['fc7_drop'], num_classes, activation=None, scope='fc8',
                          restore=restore_logits)
          end_points['fc8_drop'] = ops.dropout(end_points['fc8'], dropout_keep_prob, scope='fc8_drop')

          logits = end_points['fc8_drop']
          end_points['logits'] = logits
          end_points['predictions'] = tf.nn.softmax(logits, name='predictions')
      return logits, end_points


def vgg16_parameters(weight_decay=0.00004, stddev=0.1,
                            batch_norm_decay=0.9997, batch_norm_epsilon=0.001):
  """Yields the scope with the default parameters for inception_v3.

  Args:
    weight_decay: the weight decay for weights variables.
    stddev: standard deviation of the truncated guassian weight distribution.
    batch_norm_decay: decay for the moving average of batch_norm momentums.
    batch_norm_epsilon: small float added to variance to avoid dividing by zero.

  Yields:
    a arg_scope with the parameters needed for inception_v3.
  """
  # Set weight_decay for weights in Conv and FC layers.
  with scopes.arg_scope([ops.conv2d, ops.fc],
                        weight_decay=weight_decay):
    # Set stddev, activation and parameters for batch_norm.
    with scopes.arg_scope([ops.conv2d],
                          stddev=stddev,
                          activation=tf.nn.relu,
                          batch_norm_params={
                              'decay': batch_norm_decay,
                              'epsilon': batch_norm_epsilon}) as arg_scope:
      yield arg_scope
