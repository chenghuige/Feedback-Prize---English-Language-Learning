#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   main.py
#        \author   chenghuige  
#          \date   2022-02-10 07:02:54.233162
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gezi.common import *
sys.path.append('..')
sys.path.append('../../../../utils')
sys.path.append('../../../../third')

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["NCCL_DEBUG"] = 'WARNING'


from tensorflow import keras
from torch.utils.data import DataLoader

from transformers import (
  AutoModelForMaskedLM,
  DataCollatorForLanguageModeling,
  AutoConfig,
)
import datasets

import gezi
from gezi import tqdm
logging = gezi.logging
import melt as mt
import lele

import src
import src.eval as ev
from src import config
from src.config import *
from src.preprocess import *
from src import util

import tensorflow as tf
from absl import app, flags
FLAGS = flags.FLAGS

TEXT_COLUMN_NAME = 'essay_text'
KEYS = ['input_ids', 'attention_mask', 'token_type_ids']

from torch.utils.data import Dataset as TorchDataset

class Dataset(TorchDataset):
  def __init__(self, df, subset='valid'):
    self.subset = subset
    self.df = df
    
  def __getitem__(self, idx):
    row = dict(self.df.iloc[idx])
    if FLAGS.dynamic_line_by_line:
      row_ = {k: row[k] for k in KEYS if k in row}    
      fe = gezi.rand_range(row_, FLAGS.max_len)
    else:
      fe = row
    if FLAGS.lm_train:
      fe['label'] = 0
    return fe
    
  def __len__(self):
    return len(self.df)

def deal(ifile, lm_file, tokenizer):
  df = gezi.read_df(ifile)
  logger.info(f'loading from {ifile}')
  ic(len(df), 'fold' in df.columns)
  df_folds = None
  if (not 'fold' in df.columns) and (not FLAGS.online):
    gezi.set_fold(df, 10)
  if not 'fold' in df.columns:
    df['fold'] = -1

  if df_folds is not None:
    df = df.merge(df_folds, on='id')
  ic(df_folds, df['fold'])
  if FLAGS.lm_max_lines:
    df = df.head(FLAGS.lm_max_lines)
  df = df[['id', TEXT_COLUMN_NAME, 'fold']]
  
  rng = np.random.default_rng()
  if not FLAGS.lm_line_by_line:
    assert tokenizer.is_fast
    def tokenize_function(row):
      # seems must use words interface
      encoded = tokenizer(row[TEXT_COLUMN_NAME],
                          return_overflowing_tokens=True,
                          stride=FLAGS.lm_stride,
                          max_length=FLAGS.lm_max_len,
                          truncation=True)
      n = len(encoded['input_ids'])
      l = []
      for i in range(n):
        res = {k: encoded[k][i] for k in encoded if k in KEYS}
        for key in ['id', 'fold']:
          if key in row:
            res[key] = row[key]
        l.append(res)
      return l
  else:
    def tokenize_function(row):
      res = tokenizer(row[TEXT_COLUMN_NAME])
      if not FLAGS.dynamic_line_by_line:
        res = gezi.rand_range(res, FLAGS.lm_max_len)
      for key in ['id', 'fold']:
        if key in row:
          res[key] = row[key]
      return res

  rows = df.to_dict('records')
  l = gezi.prun_list(tokenize_function , rows, FLAGS.workers, desc='tokenize')
  if not FLAGS.lm_line_by_line:
    l = gezi.flatten_list(l)
  df = pd.DataFrame(l)
  logger.info(f'save {lm_file}')
  
  df.reset_index(drop=True).to_feather(lm_file)
  return df
  
def main(_):
  timer = gezi.Timer()
  fit = mt.fit  
  
  rank = gezi.get('RANK', 0)
  if rank != 0:
    ic.disable()
  
  if not FLAGS.lm_train:
    FLAGS.hug_inputs = True
  else:
    FLAGS.hug_inputs = False
  # FLAGS.hug_caching = True
  FLAGS.continue_pretrain = False
  # FLAGS.online = False
  FLAGS.ep = FLAGS.ep or 10
  FLAGS.bs = FLAGS.bs or 64
  FLAGS.sie = FLAGS.sie or 1
  FLAGS.lr = FLAGS.lr or 5e-5
  config.init()
  # FLAGS.acc_steps = 8
  if FLAGS.prepare:
    FLAGS.wandb = False
  FLAGS.gpus = -1
  FLAGS.rdrop_rate = 0
  FLAGS.awp_train = False
  # FLAGS.opt_fused = False
  ic(FLAGS.gpus, FLAGS.bs, FLAGS.max_len)
  backbone = FLAGS.backbone.split('/')[-1]
  # notice for lr, all other converge using 5e-5 except deberta-xlarge which also use bs 8 instead of 16 due to OOM on 4 A100 GPUs, so deberta-xlarge using 2.5e-5
  FLAGS.lr_decay_power = 1.
  
  FLAGS.mn = FLAGS.custom_backbone or backbone
  FLAGS.mn += f'.{FLAGS.max_len}'
  # FLAGS.model_dir = f'{FLAGS.root}/pretrain/base'
  ic(FLAGS.num_decay_epochs, FLAGS.ep)
  
  mt.init()
  
  ic(FLAGS.model_dir, FLAGS.mn, FLAGS.backbone, 
     backbone, FLAGS.pretrain)
  
  tokenizer = get_tokenizer(FLAGS.backbone)
  ic(tokenizer, len(tokenizer))
  
  lm_file = f'{FLAGS.root}/lm.{backbone}.fea'
  ic(lm_file)
  if os.path.exists(lm_file):
    logger.info(f'directly loading from {lm_file}')
    df = pd.read_feather(lm_file)
  else:
    assert not FLAGS.distributed, 'first use prepare mode before ddp run! ./lm_main.py --prepare'
    if not FLAGS.lm_use_type:
      # gen in jupyter/pseudo.ipynb
      ifile = f'../working/essay.csv'
    else:
      # gen in jupyter/prepare.ipynb with discourse_type
      ifile = f'../working/essay2.csv'
    ic(ifile, lm_file, FLAGS.lm_line_by_line, FLAGS.dynamic_line_by_line, FLAGS.lm_balance)
    df = deal(ifile, lm_file, tokenizer)
    if FLAGS.prepare:
      exit(0) 

  ic(df)
  
  valid_df = df[df.fold == FLAGS.fold]
  if not FLAGS.online:
    df = df[df.fold != FLAGS.fold] 
  
  ic(len(df), len(valid_df))
  
  if FLAGS.lm_train:
    # hack for custom train loop now need 1 filed of label or labels
    df['label'] = 0
    valid_df['label'] = 0
  
  if (not FLAGS.dynamic_line_by_line) and (not FLAGS.lm_custom_dataset):
    ignores = ['id', 'fold']
    ignores = [x for x in ignores if x in df.columns]
    # TODO why it comes....
    ignores += ['__index_level_0__']
    ic(ignores)
    ds = datasets.Dataset.from_pandas(df)
    ds = ds.remove_columns(ignores)
    ic(ds)
    
    valid_ds = datasets.Dataset.from_pandas(valid_df)
    valid_ds = valid_ds.remove_columns(ignores)
  else:      
    ds = Dataset(df, 'train')
    valid_ds = Dataset(valid_df, 'valid')
  
  if (not FLAGS.lm_train) or FLAGS.lm_hug_inputs:
    collate_fn = DataCollatorForLanguageModeling(
          tokenizer=tokenizer,
          mlm_probability=0.15,
          pad_to_multiple_of=None,
      )
  else:
    #for custom lm train
    collate_fn=gezi.DictPadCollate()
    
  def _get_sampler(ds, df, shuffle):
    if (not FLAGS.lm_balance) or (not shuffle):
      sampler = lele.get_sampler(ds, shuffle=shuffle)
    else:
      df['weights'] = df['input_ids'].apply(lambda x: int(len(x) / FLAGS.lm_max_len) + 1)
      ic(df['weights'].mean())
      # TODO NOTICE seems this not chaned len(sampler) and len(dl) len(ds) so each epoch may losse some points but overall ok
      sampler = lele.WeightsSampler(df.weights.values, shuffle=shuffle)
      # sampler = torch.utils.data.WeightedRandomSampler(df.weights.values, int(df.weights.sum()), replacement=False)
      if FLAGS.distributed:
        sampler = lele.DistributedSampler(sampler, FLAGS.world_size, rank, shuffle=shuffle)
    
  sampler = _get_sampler(ds, df, shuffle=True)
  kwargs = {'num_workers': FLAGS.num_workers, 'pin_memory': True, 'persistent_workers': True, 'collate_fn':collate_fn} 
  dl = torch.utils.data.DataLoader(ds, batch_size=gezi.batch_size(), sampler=sampler, **kwargs)
  ic(len(dl), len(dl.dataset), len(dl.sampler), gezi.batch_size())
  # exit(0)
  
  valid_sampler = _get_sampler(valid_ds, valid_df, shuffle=False)
  valid_dl = torch.utils.data.DataLoader(valid_ds,
                                         batch_size=gezi.eval_batch_size(),
                                         sampler=valid_sampler,
                                         **kwargs)
  ic(len(valid_dl), len(valid_dl.dataset))
  if not FLAGS.lm_train:
    model = AutoModelForMaskedLM.from_pretrained(FLAGS.backbone, trust_remote_code=True)
    model.resize_token_embeddings(len(tokenizer)) 
  else:
    # TODO why slower?  ddp ./lm_main.py 7.2it/s -> ddp ./lm_main.py --lm_train 6.7it/s
    # data collator gezi.DictPadCollate slow most likely or model？
    from src.torch.model import Model
    model = Model()
  # ic(model)
  fit(model,  
      dataset=dl,
      valid_dataset=valid_dl,
      opt_params=lele.get_opt_params(model, weight_decay=FLAGS.weight_decay),
    ) 
  
  if rank == 0:          
    if not FLAGS.lm_train:
      aconfig = AutoConfig.from_pretrained(FLAGS.backbone, trust_remote_code=True)
      aconfig.save_pretrained(FLAGS.model_dir)
      tokenizer.save_pretrained(FLAGS.model_dir)
      model.save_pretrained(FLAGS.model_dir)
    else:
      model.config.save_pretrained(FLAGS.model_dir)
      model.tokenizer.save_pretrained(FLAGS.model_dir)
      model.backbone.save_pretrained(FLAGS.model_dir)
    logger.info('tokenizer and model config saved')
    
if __name__ == '__main__':
  app.run(main)  
