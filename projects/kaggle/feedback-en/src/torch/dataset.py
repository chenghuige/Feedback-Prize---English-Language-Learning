#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   dataset.py
#        \author   chenghuige  
#          \date   2022-05-31 03:21:26.236767
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gezi.common import * 
from torch.utils.data import Dataset as TorchDataset
from src.config import *
from src.preprocess import *

class Dataset(TorchDataset):
  def __init__(self, subset='valid'):
    self.subset = subset
    self.mark = 'train' if subset in ['train', 'valid'] else 'test'
    df = get_df(self.mark)
    self.df = df
    
    df2 = None
    if (FLAGS.add_v1 or FLAGS.v1_only) and subset == 'train':
      df2 = get_df('pseudo_train')
      # if FLAGS.exclude_v2:
      #   essay_ids = set(df.essay_id)
      #   df2 = df2[~df2.essay_id.isin(essay_ids)]
      if not FLAGS.v1_only:
        if not FLAGS.trans:
          self.df = pd.concat([self.df, df2])
      else:
        self.df = df2
       
    if FLAGS.trans and subset == 'train':
      self.df = get_df(FLAGS.trans)
      ic(len(self.df), set(self.df.fold.values))
      if df2 is not None:
        self.df = pd.concat([self.df, df2])
        
    ic(subset, len(self.df), len(df), set(self.df.fold.values))
    
    if (not FLAGS.online) and (not FLAGS.use_v1):
      if subset == 'valid':
        self.df = df[df.fold==FLAGS.fold]
      elif subset == 'train':
        self.df = self.df[self.df.fold!=FLAGS.fold]
        # if not FLAGS.exclude_v2:
        #   essay_ids = set(df[df.fold==FLAGS.fold].essay_id)
        #   self.df = self.df[~self.df.essay_id.isin(essay_ids)]
    
    ic(subset, len(self.df), set(self.df.fold.values))
    self.rng = np.random.default_rng(FLAGS.aug_seed)
    idx = self.rng.integers(len(self))
    # NOTICE! ic(fe) might hang for some idx which might due to fe['input_tokens'] too large
    if gezi.in_kaggle():
      idx = 0
    else:
      self.show(idx)

  def show(self, idx=0):
    fe = self[idx]
    assert 'input_ids' in fe
    tokenizer = get_tokenizer(FLAGS.backbone)
    fe['input_tokens'] = ''.join(tokenizer.convert_ids_to_tokens(fe['input_ids']))
    if gezi.in_kaggle():
      fe['input_tokens'] = fe['input_tokens'][:1024]
    fe['num_input_tokens'] = len(fe['input_ids'])
    if 'token_type_ids' in fe:
      fe['token_type_ids_str'] = ''.join(map(str, fe['token_type_ids']))
      if gezi.in_kaggle():
        fe['token_type_ids_str'] = fe['token_type_ids_str'][:1024]
    keys = [
      'essay_id', 'discourse_id',	'discourse_type',	'discourse_effectiveness',
      'input_tokens', 'token_type_ids_str', 'num_input_tokens',
      ]
    fe = OrderedDict({k: fe[k] for k in keys if k in fe})
    logger.info(f'fe_show subset:{self.subset} idx:{idx} fe:{fe}')

  def __getitem__(self, idx):
    row = dict(self.df.iloc[idx])   
    fe = parse_example(row, subset=self.subset, rng=self.rng)
    return fe
    
  def __len__(self):
    return len(self.df)
  