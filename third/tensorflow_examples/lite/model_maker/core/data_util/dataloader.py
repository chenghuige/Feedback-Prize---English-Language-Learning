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
"""Common Dataset used for tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class DataLoader(object):
  """This class provides generic utilities for loading customized domain data that will be used later in model retraining.

  For different ML problems or tasks, such as image classification, text
  classification etc., a subclass is provided to handle task-specific data
  loading requirements.
  """

  def __init__(self, dataset, size):
    """Init function for class `DataLoader`.

    In most cases, one should use helper functions like `from_folder` to create
    an instance of this class.

    Args:
      dataset: A tf.data.Dataset object that contains a potentially large set of
        elements, where each element is a pair of (input_data, target). The
        `input_data` means the raw input data, like an image, a text etc., while
        the `target` means some ground truth of the raw input data, such as the
        classification label of the image etc.
      size: The size of the dataset. tf.data.Dataset donesn't support a function
        to get the length directly since it's lazy-loaded and may be infinite.
    """
    self.dataset = dataset
    self.size = size

  def __len__(self):
    return self.size

  def split(self, fraction):
    """Splits dataset into two sub-datasets with the given fraction.

    Primarily used for splitting the data set into training and testing sets.

    Args:
      fraction: float, demonstrates the fraction of the first returned
        subdataset in the original data.

    Returns:
      The splitted two sub dataset.
    """
    ds = self.dataset

    train_size = int(self.size * fraction)
    trainset = DataLoader(ds.take(train_size), train_size)

    test_size = self.size - train_size
    testset = DataLoader(ds.skip(test_size), test_size)

    return trainset, testset
