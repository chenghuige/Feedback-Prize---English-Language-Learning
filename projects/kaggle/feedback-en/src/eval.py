#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   eval.py
#        \author   chenghuige  
#          \date   2022-05-11 11:12:10.236762
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gezi.common import * 
from src.config import *
from src.postprocess import *
from sklearn.metrics import log_loss

def mcrmse(labels: np.ndarray, preds: np.ndarray) -> float:
  # ic(labels, preds, labels.shape, preds.shape)
  colwise_rmse = np.sqrt(np.mean((labels - preds) ** 2, axis=0))
  mean_rmse = np.mean(colwise_rmse)
  return mean_rmse

def calc_metric(y_true, y_pred):
  return mcrmse(y_true, y_pred)

def calc_metrics(y_true, y_pred):
  if FLAGS.train_version > 0:
    return {'score': calc_metric(y_true, y_pred)}
  res = OrderedDict()
  res['score'] = 0.
  weights = get_weights(reshape=False)
  for i, target in enumerate(TARGETS):
    # if weights[i] == 0:
    #   continue 
    res[f'score/{target}'] = calc_metric(y_true[:, i], y_pred[:, i])
    res['score'] += res[f'score/{target}']
  res['score'] /= len(TARGETS)
    
  cls_pred = pred2cls(y_pred)
  res['score/cls'] = calc_metric(y_true, cls_pred)
  
  res['acc'] = (y_true == cls_pred).mean()
  for i, target in enumerate(TARGETS):
    res[f'acc/{target}'] = (y_true[:, i] == cls_pred[:, i]).mean()
  
  y_true_mean = y_true.mean(-1, keepdims=True)
  delta = abs(y_true - y_true_mean)
  max_idx = np.argmax(delta, -1)[:,np.newaxis]
  y_true_max = np.take_along_axis(y_true, max_idx, 1)
  y_pred_max = np.take_along_axis(y_pred, max_idx, 1)
  res['score/max'] = calc_metric(y_true_max, y_pred_max)
  res['acc/max'] = (y_true_max == pred2cls(y_pred_max)).mean()
  min_idx = np.argmin(delta, -1)[:,np.newaxis]
  y_true_min = np.take_along_axis(y_true, min_idx, 1)
  y_pred_min = np.take_along_axis(y_pred, min_idx, 1)
  res['score/min'] = calc_metric(y_true_min, y_pred_min)
  res['acc/min'] = (y_true_min == pred2cls(y_pred_min)).mean()
  
  y_pred_mean = y_pred.mean(-1, keepdims=True)
  delta = abs(y_pred - y_pred_mean)
  max_idx = np.argmax(delta, -1)[:,np.newaxis]
  y_true_max = np.take_along_axis(y_true, max_idx, 1)
  y_pred_max = np.take_along_axis(y_pred, max_idx, 1)
  res['score/max2'] = calc_metric(y_true_max, y_pred_max)
  min_idx = np.argmin(delta, -1)[:,np.newaxis]
  y_true_min = np.take_along_axis(y_true, min_idx, 1)
  y_pred_min = np.take_along_axis(y_pred, min_idx, 1)
  res['score/min2'] = calc_metric(y_true_min, y_pred_min)
  
  
  res['count'] = len(y_true)
  return res

def evaluate(y_true, y_pred, x, other, is_last=False):
  res = {}
  
  eval_dict = gezi.get('eval_dict')
  if eval_dict:
    x.update(eval_dict)
    
  x.update(other)

  gezi.set('eval:x', x)
  
  y_pred = to_pred(x['pred'])
  # y_true = x['label']
  y_true = to_pred(x['label'])
  
  # ic(y_true, y_pred, y_true.shape, y_pred.shape)
  
  res.update(calc_metrics(y_true, y_pred))

  df = pd.DataFrame({
    'n_words': x['n_words'],
    'label': list(y_true),
    'pred': list(y_pred),
  })
    
  df_ = df[df.n_words < 400]
  res['score/400'] = calc_metric(np.vstack(df_['label'].values), np.vstack(df_['pred'].values))
  
  df_ = df[(df.n_words >= 400) & (df.n_words < 800)]
  res['score/400-800'] = calc_metric(np.vstack(df_['label'].values), np.vstack(df_['pred'].values))
  
  df_ = df[df.n_words >= 800]
  res['score/800+'] = calc_metric(np.vstack(df_['label'].values), np.vstack(df_['pred'].values))

  if FLAGS.train_version == 0:
    weights = get_weights(reshape=False)
    if weights.min() > 0 and res['score'] > 0.7:
      logging.error(res)
      gezi.remove_dir(FLAGS.model_dir)
      os.system('kill-match.sh main')
      exit(-1)
    
    # for i, target in enumerate(TARGETS):
    #   if weights[i] > 0 and res[f'score/{target}'] > 0.7:
    #     logging.error(target, res[f'score/{target}'])
    #     gezi.remove_dir(FLAGS.model_dir)
    #     os.system('kill-match.sh main')
    #     exit(-1)

  return res

def valid_write(x, label, predicts, ofile, others={}):
  ofile = f'{FLAGS.model_dir}/valid.csv'
  write_result(x, predicts, ofile, others, is_infer=False)

def infer_write(x, predicts, ofile, others={}):
  ofile = f'{FLAGS.model_dir}/submission.csv'
  write_result(x, predicts, ofile, others, is_infer=True)
  
def to_df(x):
  predicts = np.asarray(x['pred'])
  predicts = to_pred(predicts)
  m = OrderedDict({
    'text_id': x['text_id'],
  })
  if FLAGS.train_version > 0:
    m['score'] = predicts
  else:
    for i, target in enumerate(TARGETS):
      m[target] = list(predicts[:,i])

  df = pd.DataFrame(m)
  return df
      
def write_result(x, predicts, ofile, others={}, is_infer=False, need_normalize=True):
  if is_infer:
    m = gezi.get('infer_dict')
  else:
    m = gezi.get('eval_dict')
    
  if m:
    x.update(m)
  x.update(others)
  df = to_df(x)
  if not is_infer:
    y_true = to_pred(x['label'])
    df['label'] = list(y_true)
  df = df.sort_values('text_id')
  ic(df)
  df.to_csv(ofile, index=False)
  