# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Configurations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_examples.lite.model_maker.core import compat

DEFAULT_QUANTIZATION_STEPS = 2000


def _get_representative_dataset_gen(dataset, num_steps):
  """Gets the function that generates representative dataset for quantized."""

  def representative_dataset_gen():
    """Generates representative dataset for quantized."""
    if compat.get_tf_behavior() == 2:
      for image, _ in dataset.take(num_steps):
        yield [image]
    else:
      iterator = tf.compat.v1.data.make_initializable_iterator(
          dataset.take(num_steps))
      next_element = iterator.get_next()
      with tf.compat.v1.Session() as sess:
        sess.run(iterator.initializer)
        while True:
          try:
            image, _ = sess.run(next_element)
            yield [image]
          except tf.errors.OutOfRangeError:
            break

  return representative_dataset_gen


class QuantizationConfig(object):
  """Configuration for post-training quantization.

  Refer to
  https://www.tensorflow.org/lite/performance/post_training_quantization
  for different post-training quantization options.
  """

  def __init__(
      self,
      optimizations=None,
      representative_data=None,
      quantization_steps=None,
      inference_input_type=None,
      inference_output_type=None,
      supported_ops=None,
      supported_types=None,
      _experimental_new_quantizer=None,
  ):
    """Constructs QuantizationConfig.

    Args:
      optimizations: A list of optimizations to apply when converting the model.
        If not set, use `[Optimize.DEFAULT]` by default.
      representative_data: Representative data used for post-training
        quantization.
      quantization_steps: Number of post-training quantization calibration steps
        to run.
      inference_input_type: Target data type of real-number input arrays. Allows
        for a different type for input arrays. Defaults to None. If set, must be
        be `{tf.float32, tf.uint8, tf.int8}`.
      inference_output_type: Target data type of real-number output arrays.
        Allows for a different type for output arrays. Defaults to None. If set,
        must be `{tf.float32, tf.uint8, tf.int8}`.
      supported_ops: Set of OpsSet options supported by the device. Used to Set
        converter.target_spec.supported_ops.
      supported_types: List of types for constant values on the target device.
        Supported values are types exported by lite.constants. Frequently, an
        optimization choice is driven by the most compact (i.e. smallest) type
        in this list (default [constants.FLOAT]).
      _experimental_new_quantizer: Whether to enable experimental new quantizer.
    """

    if optimizations is None:
      optimizations = [tf.lite.Optimize.DEFAULT]
    if not isinstance(optimizations, list):
      optimizations = [optimizations]
    self.optimizations = optimizations

    self.representative_data = representative_data
    if self.representative_data is not None and quantization_steps is None:
      quantization_steps = DEFAULT_QUANTIZATION_STEPS
    self.quantization_steps = quantization_steps

    self.inference_input_type = inference_input_type
    self.inference_output_type = inference_output_type

    if supported_ops is not None and not isinstance(supported_ops, list):
      supported_ops = [supported_ops]
    self.supported_ops = supported_ops

    if supported_types is not None and not isinstance(supported_types, list):
      supported_types = [supported_types]
    self.supported_types = supported_types

    self._experimental_new_quantizer = _experimental_new_quantizer

  @classmethod
  def create_dynamic_range_quantization(cls,
                                        optimizations=tf.lite.Optimize.DEFAULT):
    """Creates configuration for dynamic range quantization."""
    return QuantizationConfig(optimizations)

  @classmethod
  def create_full_integer_quantization(
      cls,
      representative_data,
      quantization_steps=DEFAULT_QUANTIZATION_STEPS,
      optimizations=tf.lite.Optimize.DEFAULT,
      inference_input_type=tf.uint8,
      inference_output_type=tf.uint8,
      is_integer_only=False):
    """Creates configuration for full integer quantization.

    Args:
      representative_data: Representative data used for post-training
        quantization.
      quantization_steps: Number of post-training quantization calibration steps
        to run.
      optimizations: A list of optimizations to apply when converting the model.
        If not set, use `[Optimize.DEFAULT]` by default.
      inference_input_type: Target data type of real-number input arrays. Used
        only when `is_integer_only` is True. Must be in `{tf.uint8, tf.int8}`.
      inference_output_type: Target data type of real-number output arrays. Used
        only when `is_integer_only` is True. Must be in `{tf.uint8, tf.int8}`.
      is_integer_only: If True, enforces full integer quantization for all ops
        including the input and output. If False, uses integer with float
        fallback (using default float input/output) that mean to fully integer
        quantize a model, but use float operators when they don't have an
        integer implementation.

    Returns:
      QuantizationConfig.
    """
    if not is_integer_only:
      return QuantizationConfig(
          optimizations,
          representative_data=representative_data,
          quantization_steps=quantization_steps)
    else:
      if inference_input_type not in [tf.uint8, tf.int8]:
        raise ValueError('For integer only quantization, '
                         '`inference_input_type` '
                         'should be tf.uint8 or tf.int8.')
      if inference_output_type not in [tf.uint8, tf.int8]:
        raise ValueError('For integer only quantization, '
                         '`inference_output_type` '
                         'should be tf.uint8 or tf.int8.')

      return QuantizationConfig(
          optimizations,
          representative_data=representative_data,
          quantization_steps=quantization_steps,
          inference_input_type=inference_input_type,
          inference_output_type=inference_output_type,
          supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8])

  @classmethod
  def create_float16_quantization(cls, optimizations=tf.lite.Optimize.DEFAULT):
    """Creates configuration for float16 quantization."""
    return QuantizationConfig(optimizations, supported_types=[tf.float16])

  def get_converter_with_quantization(self, converter, gen_dataset_fn=None):
    """Gets TFLite converter with settings for quantization."""
    converter.optimizations = self.optimizations

    if self.representative_data is not None:
      if gen_dataset_fn is None:
        raise ValueError('Must provide "gen_dataset_fn" when'
                         '"representative_data" is not None.')
      ds = gen_dataset_fn(
          self.representative_data, batch_size=1, is_training=False)
      converter.representative_dataset = tf.lite.RepresentativeDataset(
          _get_representative_dataset_gen(ds, self.quantization_steps))

    if self.inference_input_type:
      converter.inference_input_type = self.inference_input_type
    if self.inference_output_type:
      converter.inference_output_type = self.inference_output_type
    if self.supported_ops:
      converter.target_spec.supported_ops = self.supported_ops
    if self.supported_types:
      converter.target_spec.supported_types = self.supported_types

    if self._experimental_new_quantizer is not None:
      converter._experimental_new_quantizer = self._experimental_new_quantizer  # pylint: disable=protected-access
    return converter
