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
"""Library to retrain image classifier models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import tempfile

import tensorflow.compat.v2 as tf
from tensorflow_examples.lite.model_maker.core.optimization import warmup
from tensorflow_hub.tools.make_image_classifier import make_image_classifier_lib as hub_lib

DEFAULT_DECAY_SAMPLES = 10000 * 256
DEFAULT_WARMUP_EPOCHS = 2


def add_params(hparams, **kwargs):
  param_dict = {k: v for k, v in kwargs.items() if v is not None}

  return hparams._replace(**param_dict)


class HParams(
    collections.namedtuple(
        "HParams", hub_lib.HParams._fields + ("warmup_steps", "model_dir"))):
  """The hyperparameters for make_image_classifier.

  train_epochs: Training will do this many iterations over the dataset.
  do_fine_tuning: If true, the Hub module is trained together with the
    classification layer on top.
  batch_size: Each training step samples a batch of this many images.
  learning_rate: Base learning rate when train batch size is 256. Linear to the
    batch size.
  dropout_rate: The fraction of the input units to drop, used in dropout layer.
  warmup_steps: Number of warmup steps for warmup schedule on learning rate.
  model_dir: The location of the model checkpoint files.
  """

  @classmethod
  def get_hparams(cls, **kwargs):
    """Gets the hyperparameters for `train_image_classifier_lib`."""
    hparams = get_default_hparams()
    return add_params(hparams, **kwargs)


def get_default_hparams():
  """Returns a fresh HParams object initialized to default values."""
  default_hub_hparams = hub_lib.get_default_hparams()
  as_dict = default_hub_hparams._asdict()
  as_dict.update(
      train_epochs=10,
      do_fine_tuning=False,
      batch_size=64,
      learning_rate=0.004,
      dropout_rate=0.2,
      warmup_steps=None,
      model_dir=tempfile.mkdtemp(),
  )
  default_hparams = HParams(**as_dict)
  return default_hparams


def create_optimizer(init_lr, num_decay_steps, num_warmup_steps):
  """Creates an optimizer with learning rate schedule."""
  # Leverages cosine decay of the learning rate.
  learning_rate_fn = tf.keras.experimental.CosineDecay(
      initial_learning_rate=init_lr, decay_steps=num_decay_steps, alpha=0.0)
  if num_warmup_steps:
    learning_rate_fn = warmup.WarmUp(
        initial_learning_rate=init_lr,
        decay_schedule_fn=learning_rate_fn,
        warmup_steps=num_warmup_steps)
  optimizer = tf.keras.optimizers.RMSprop(
      learning_rate=learning_rate_fn, rho=0.9, momentum=0.9, epsilon=0.001)

  return optimizer


def train_model(model, hparams, train_data_and_size, validation_data_and_size):
  """Trains model with the given data and hyperparameters.

  Args:
    model: The tf.keras.Model from _build_model().
    hparams: A namedtuple of hyperparameters. This function expects
      .train_epochs: a Python integer with the number of passes over the
        training dataset;
      .learning_rate: a Python float forwarded to the optimizer; Base learning
        rate when train batch size is 256. Linear to the batch size;
      .batch_size: a Python integer, the number of examples returned by each
        call to the generators;
      .warmup_steps: a Python integer, the number of warmup steps for warmup
        schedule on learning rate. If None, default warmup_steps is used;
      .model_dir: a Python string, the location of the model checkpoint files.
    train_data_and_size: A (data, size) tuple in which data is training data to
      be fed in tf.keras.Model.fit(), size is a Python integer with the numbers
      of training.
    validation_data_and_size: A (data, size) tuple in which data is validation
      data to be fed in tf.keras.Model.fit(), size is a Python integer with the
      numbers of validation.

  Returns:
    The tf.keras.callbacks.History object returned by tf.keras.Model.fit().
  """
  train_data, train_size = train_data_and_size
  validation_data, validation_size = validation_data_and_size

  steps_per_epoch = train_size // hparams.batch_size
  validation_steps = validation_size // hparams.batch_size

  # Learning rate is linear to batch size.
  learning_rate = hparams.learning_rate * hparams.batch_size / 256

  # Gets decay steps.
  total_training_steps = steps_per_epoch * hparams.train_epochs
  default_decay_steps = DEFAULT_DECAY_SAMPLES // hparams.batch_size
  decay_steps = max(total_training_steps, default_decay_steps)

  warmup_steps = hparams.warmup_steps
  if warmup_steps is None:
    warmup_steps = DEFAULT_WARMUP_EPOCHS * steps_per_epoch
  optimizer = create_optimizer(learning_rate, decay_steps, warmup_steps)

  loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
  model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

  summary_dir = os.path.join(hparams.model_dir, "summaries")
  summary_callback = tf.keras.callbacks.TensorBoard(summary_dir)
  # Save checkpoint every 20 epochs.
  checkpoint_path = os.path.join(hparams.model_dir, "checkpoint")
  checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
      checkpoint_path, save_weights_only=True, period=20)

  # Trains the models.
  return model.fit(
      train_data,
      epochs=hparams.train_epochs,
      steps_per_epoch=steps_per_epoch,
      validation_data=validation_data,
      validation_steps=validation_steps,
      callbacks=[summary_callback, checkpoint_callback])
