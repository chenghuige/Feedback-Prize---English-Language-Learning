#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   postprocess.py
#        \author   chenghuige  
#          \date   2022-05-11 11:17:37.027553
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gezi.common import * 
from src.config import *
  
def to_pred(pred):
  # if pred.shape[1] > NUM_TARGETS:
    # pred = pred[:,2:]
  if FLAGS.method == 'cls' and pred.shape[-1] == NUM_LABELS:
    pred = gezi.softmax(pred, -1)
    pred = (pred * LABELS.reshape(1, 1, NUM_LABELS)).sum(-1)
  else:
    if FLAGS.norm_target:
      pred = pred * 4 + 1
  pred = np.clip(pred, 1.0, 5.0)
  return pred

def pred2cls_(pred):
  label = np.asarray([1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
  return label[np.argmin(abs(label - pred))]
  
def pred2cls(pred):
  pred_ = pred.reshape(-1)
  pred_ = np.asarray([pred2cls_(x) for x in pred_])
  return pred_.reshape(pred.shape)
