#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   loss.py
#        \author   chenghuige  
#          \date   2022-05-11 11:13:01.390146
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gezi.common import * 
from torch import nn
from src.config import *
from src.preprocess import get_tokenizer

def calc_loss(res, y, x, step=None, epoch=None, training=None):
  scalars = {}

  y = x['label'].float()
  y_ = res[FLAGS.pred_key].float()
  
  num_targets = NUM_TARGETS
  if FLAGS.train_version > 0:
    num_targets = 1
    
  # if FLAGS.aux_loss:
  # y = torch.cat([y.mean(-1, keepdim=True), y.min(-1, keepdim=True)[0], y.max(-1, keepdim=True)[0], y], 1)
  # y = torch.cat([y.min(-1, keepdim=True)[0], y.max(-1, keepdim=True)[0], y], 1)
  # y_ = torch.cat([y_, y_.max(-1, keepdim=True)[0], y_.min(-1, keepdim=True)[0]], 1)
  # ic(y.shape, y_.shape)
  # exit(0)
  # num_targets += 2
    
  reduction = FLAGS.loss_reduction
  loss = 0.

  if FLAGS.lm_train:
    loss_obj = nn.CrossEntropyLoss(label_smoothing=FLAGS.label_smoothing, reduction=reduction)
    # ic(res['pred'].shape, res['label'].shape, get_tokenizer(FLAGS.backbone).vocab_size, len(get_tokenizer(FLAGS.backbone)))
    lm_loss = loss_obj(res['pred'].contiguous().view(-1, len(get_tokenizer(FLAGS.backbone))), res['label'].contiguous().view(-1))
    scalars['loss/lm'] = lm_loss.item()
    loss += lm_loss
    return loss

  if FLAGS.loss_fn == 'ce':
    loss_obj = nn.CrossEntropyLoss(reduction=reduction, label_smoothing=FLAGS.label_smoothing)  
  elif FLAGS.loss_fn == 'focal':
    loss_obj = lele.losses.FocalLoss(alpha=FLAGS.focal_alpha, gamma=FLAGS.focal_gamma)
  elif FLAGS.loss_fn == 'mse':
    loss_obj = nn.MSELoss(reduction=reduction)
  elif FLAGS.loss_fn == 'mae':
    loss_obj = nn.L1Loss(reduction=reduction)
  elif FLAGS.loss_fn == 'sl1':
    loss_obj = nn.SmoothL1Loss(reduction=reduction)
  elif FLAGS.loss_fn == 'rmse':
    loss_obj = lele.losses.RMSELoss(reduction=reduction)
  elif FLAGS.loss_fn == 'mcrmse':
    loss_obj = lele.losses.MCRMSELoss()
  else:
    raise ValueError('not supported loss fn', FLAGS.loss_fn)
  
  # ic(y, y_, y.shape, y_.shape)
  if not FLAGS.soft_label:
    if not 'labels' in x:
      if not 'logits_list' in res:
        if not FLAGS.method == 'cls':
          base_loss = loss_obj(y_.view(-1, num_targets), y.view(-1, num_targets))
        else:
          base_loss = loss_obj(y_.view(-1, NUM_LABELS), y.view(-1).long())
      else:
        loss_list = [loss_obj(logits.view(-1, num_targets), y.view(-1, num_targets)) for logits in res['logits_list']]
        base_loss = torch.stack(loss_list, 0).mean()
    else:
      base_loss = loss_obj(y_.view(-1, num_targets), x['labels'].float().view(-1, num_targets))
      # if not training:
      #   ic(y_, x['labels'], base_loss.item())
  else:
    y = x['soft_label']
    base_loss = loss_obj(y_.view(-1, num_targets), y.view(-1, num_targets))
  
  if reduction == 'none':
    if FLAGS.max_loss:
      if not FLAGS.add_max_loss:
        base_loss = base_loss.max(-1)[0].mean()
      else:
        base_loss = (base_loss.mean() + base_loss.max(-1)[0].mean()) / 2.
    else:
      mask = (y != -100).int()
      # if (FLAGS.target is not None) and FLAGS.nontarget_weight:
      weights = get_weights() if not 'weights' in x else x['weights']
      base_loss = base_loss * torch.as_tensor(weights, dtype=base_loss.dtype, device=base_loss.device)
      base_loss = lele.masked_mean(base_loss, mask)
   
  scalars['loss/base'] = base_loss.item()
  base_loss *= FLAGS.base_loss_rate
  loss += base_loss
  
  if FLAGS.rank_loss_rate > 0:
    pred = y_.view(-1)
    label = y.view(-1)
    # ic(pred, label)
    mask = (label != -100).bool()
    label = label.masked_select(mask)
    pred = pred.masked_select(mask)
    if FLAGS.rank_loss == 'pearson':
      rank_loss = lele.losses.pearson_loss(pred, label.float())
    elif FLAGS.rank_loss == 'spearman':
      rank_loss = lele.losses.spearman_loss(pred, label.float())
    # elif FLAGS.rank_loss == 'auc':
    #   from torchmetrics.functional import auc
    #   rank_loss = 1. - auc(pred, label.float(), reorder=True)
    elif FLAGS.rank_loss == 'label':
      from torchmetrics.functional import label_ranking_loss
      rank_loss = label_ranking_loss(y_, y.float())
    else:
      raise ValueError(FLAGS.rank_loss)
    scalars['loss/rank'] = rank_loss.item()
    rank_loss *= FLAGS.rank_loss_rate
    loss += rank_loss 
    
  if FLAGS.aux_loss_rate > 0:
    aux_y = y - y.mean(-1, keepdim=True)
    aux_y_ = y_ - y_.mean(-1, keepdim=True)
    if FLAGS.aux_abs:
      aux_y = torch.abs(aux_y)
      aux_y_ = torch.abs(aux_y_)
    aux_loss = loss_obj(aux_y_.view(-1, num_targets), aux_y.view(-1, num_targets))
    scalars['loss/aux'] = aux_loss.item()
    aux_loss *= FLAGS.aux_loss_rate
    loss += aux_loss
  
  loss *= FLAGS.loss_scale
  
  if FLAGS.rdrop_rate > 0:
    ## FIXME TODO for electra not work so [electra/roberta] why? not just use awp no rdrop, deberta/bart all ok
    # RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.cuda.LongTensor [1, 229]] is at version 3;
    # expected version 2 instead. Hint: enable anomaly detection to find the operation that failed to compute its gradient, with torch.autograd.set_detect_anomaly(True).          
    def rdrop_loss(p, q, mask=None):
      rloss = 0.
      if not FLAGS.listwise:
        rloss += lele.losses.compute_kl_loss(p['pred'], q['pred'])
      else:
        rloss += lele.losses.compute_kl_loss(p['pred'], q['pred'], mask=mask)
      return rloss
    gezi.set('rdrop_loss_fn', lambda p, q: rdrop_loss(p, q, (x['label'] != -100).unsqueeze(-1).int()))
          
  lele.update_scalars(scalars, decay=FLAGS.loss_decay, training=training)
  return loss
  
