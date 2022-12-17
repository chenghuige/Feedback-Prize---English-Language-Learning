# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""A class of Decoders that may sample to generate the next input.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from tensorflow.contrib.seq2seq.python.ops import decoder
#from tensorflow.contrib.seq2seq.python.ops import helper as helper_py
from melt.seq2seq import helper as helper_py
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base as layers_base
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.util import nest
from tensorflow.python.ops import array_ops


import tensorflow as tf 

__all__ = [
    #"LogProbsDecoderOutput",
    "LogProbsDecoderState",
    "LogProbsDecoder",
]

##TODO why wrong .. ValueError: Inconsistent shapes: saw (?, 10148) but expected (?,) (and infer_shape=True)
##Object was never used (type <class 'tensorflow.python.ops.tensor_array_ops.TensorArray'>):
##<tensorflow.python.ops.tensor_array_ops.TensorArray object at 0x9c97a10>

class LogProbsDecoderOutput(
    collections.namedtuple("LogProbsDecoderOutput", ("rnn_output", "sample_id", "log_probs"))):
  pass

# class BasicDecoderOutput(
#     collections.namedtuple("BasicDecoderOutput", ("rnn_output", "sample_id"))):
#   pass

class LogProbsDecoderState(
    collections.namedtuple("LogProbsDecoderState", ("cell_state", "log_probs"))):
  pass

class LogProbsDecoder(decoder.Decoder):
  """Basic sampling decoder."""

  def __init__(self, cell, helper, initial_state, vocab_size=None, output_fn=None):
    """Initialize BasicDecoder.

    Args:
      cell: An `RNNCell` instance.
      helper: A `Helper` instance.
      initial_state: A (possibly nested tuple of...) tensors and TensorArrays.
        The initial state of the RNNCell.
      output_fn: (Optional) An instance of `tf.layers.Layer`, i.e.,
        `tf.layers.Dense`.  Optional layer to apply to the RNN output prior
        to storing the result or sampling.

    Raises:
      TypeError: if `cell`, `helper` or `output_fn` have an incorrect type.
    """
    if not isinstance(cell, rnn_cell_impl.RNNCell):
      raise TypeError("cell must be an RNNCell, received: %s" % type(cell))
    #if not isinstance(helper, helper_py.Helper):
    #  raise TypeError("helper must be a Helper, received: %s" % type(helper))

    self._cell = cell
    self._helper = helper
    self._initial_state = initial_state

    self._output_fn = None
    if vocab_size is not None:
      if output_fn is not None:
        self._output_fn = output_fn
      else:
        self._output_fn = lambda cell_output: tf.contrib.layers.fully_connected(
          inputs=cell_output, 
          num_outputs=vocab_size, 
          activation_fn=None)

    self._vocab_size = vocab_size

    self._output_size = self._vocab_size if self._vocab_size is not None else self._cell.output_size

  @property
  def batch_size(self):
    return self._helper.batch_size

  @property
  def output_size(self):
    return LogProbsDecoderOutput(
        rnn_output=self._output_size,
        sample_id=tf.TensorShape([]),
        log_probs=tf.TensorShape([]))

  @property
  def output_dtype(self):
    return LogProbsDecoderOutput(
        rnn_output=tf.float32,
        sample_id=tf.int32,
        log_probs=tf.float32)

  # @property
  # def output_size(self):
  #   return BasicDecoderOutput(
  #       rnn_output=self._output_size,
  #       sample_id=tf.TensorShape([]))

  # @property
  # def output_dtype(self):
  #   return BasicDecoderOutput(
  #       rnn_output=tf.float32,
  #       sample_id=tf.int32)

  def initialize(self, name=None):
    """Initialize the decoder.

    Args:
      name: Name scope for any created operations.

    Returns:
      `(finished, first_inputs, initial_state)`.
    """
    initial_state = LogProbsDecoderState(
        cell_state=self._initial_state,
        log_probs=array_ops.zeros(
            [self.batch_size,],
            dtype=nest.flatten(self._initial_state)[0].dtype))
    return self._helper.initialize() + (initial_state,)

  def step(self, time, inputs, state, name=None):
    """Perform a decoding step.

    Args:
      time: scalar `int32` tensor.
      inputs: A (structure of) input tensors.
      state: A (structure of) state tensors and TensorArrays.
      name: Name scope for any created operations.

    Returns:
      `(outputs, next_state, next_inputs, finished)`.
    """
    with ops.name_scope(name, "LogProbsDecoderStep", (time, inputs, state)):
      cell_outputs, cell_state = self._cell(inputs, state.cell_state)
      if self._output_fn is not None:
        try:
          cell_outputs = self._output_fn(cell_outputs)
        except Exception:
          try:
            cell_outputs = self._output_fn(cell_outputs, cell_state)
          except Exception:
            cell_outputs = self._output_fn(time, cell_outputs, cell_state)

      sample_ids, log_probs = self._helper.sample(
          time=time, outputs=cell_outputs, state=cell_state)

      next_state = LogProbsDecoderState(cell_state, state.log_probs + log_probs)
      (finished, next_inputs, next_state) = self._helper.next_inputs(
          time=time,
          outputs=cell_outputs,
          state=next_state,
          sample_ids=sample_ids)
    
    outputs = LogProbsDecoderOutput(cell_outputs, sample_ids, log_probs) 
    #outputs = BasicDecoderOutput(cell_outputs, sample_ids)
    return (outputs, next_state, next_inputs, finished)

