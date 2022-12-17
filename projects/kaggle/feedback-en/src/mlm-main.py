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

def main(_):
  timer = gezi.Timer()
  fit = mt.fit  
  
  rank = gezi.get('RANK', 0)
  if rank != 0:
    ic.disable()
  
  FLAGS.caching = True
  FLAGS.hug_inputs = True
  FLAGS.continue_pretrain = False
  FLAGS.online = True
  FLAGS.ep = FLAGS.ep or 10
  config.init()
  FLAGS.gpus = -1
  FLAGS.sie = 0
  FLAGS.rdrop_rate = 0
  FLAGS.awp_train = False
  FLAGS.lr = 5e-5
  # FLAGS.opt_fused = False
  ic(FLAGS.gpus, FLAGS.bs, FLAGS.max_len)
  backbone = FLAGS.backbone.split('/')[-1]
  # notice for lr, all other converge using 5e-5 except deberta-xlarge which also use bs 8 instead of 16 due to OOM on 4 A100 GPUs, so deberta-xlarge using 2.5e-5
  FLAGS.lr_decay_power = 1.
  
  FLAGS.mn = FLAGS.custom_backbone or backbone
  FLAGS.model_dir = f'{FLAGS.root}/pretrain/base'
  
  mt.init()
  
  ic(FLAGS.model_dir, FLAGS.mn, FLAGS.backbone, backbone)
  tokenizer = get_tokenizer(FLAGS.backbone)
  ic(tokenizer, len(tokenizer))
  
  text_column_name = 'text'
  
  dfs = []
  if not FLAGS.lm_use_type:
    # gen in jupyter/pseudo.ipynb
    ifile = f'../working/essay.csv'
  else:
    # gen in jupyter/prepare.ipynb with discourse_type
    ifile = f'../working/essay2.csv'
  df = pd.read_csv(ifile)
  if not 'fold' in df.columns:
    gezi.set_fold(df, 10)
  df[text_column_name] = df['essay_text']
  df = df[[text_column_name, 'fold']]
  dfs.append(df)
  
  df = pd.concat(dfs)
  
  # df = df.head(1000)
  ds = datasets.Dataset.from_pandas(df)
  
  ic(ds, ds[-1])
  num_proc = 32 if FLAGS.pymp else 1
  gezi.try_mkdir(f'{FLAGS.root}/cache')
  
  def preprocess(text, method=None):
    return text.replace('\n', BR)
  
  # cache_file = f'{FLAGS.root}/cache/{FLAGS.mn}_mlm.arrow'
  ds = ds.map(lambda example: {text_column_name: preprocess(example[text_column_name])}, 
              remove_columns=[x for x in ds.column_names if x != 'fold'],
              batched=False, 
              num_proc=num_proc, 
              # cache_file_name=cache_file,
              )
  ic(ds, ds[-1])

  def tokenize_function(examples):
      # Remove empty lines
      examples[text_column_name] = [
          line for line in examples[text_column_name] if line and len(line) > 0 and not line.isspace()
      ]
      return tokenizer(
          examples[text_column_name],
          padding=False,
          truncation=True,
          max_length=FLAGS.max_len,
          return_special_tokens_mask=True,
      )

  ds = ds.map(
      tokenize_function,
      batched=True,
      num_proc=num_proc,
      remove_columns=[text_column_name],
      desc="Running tokenizer on dataset line_by_line",
  )
  ic(ds)
  
  collate_fn = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=0.15,
        pad_to_multiple_of=None,
    )

  valid_ds = ds.filter(lambda x: x['fold'] == 0, num_proc=FLAGS.workers)
  ds = ds.filter(lambda x: x['fold'] != 0, num_proc=FLAGS.workers)
  
  valid_ds = valid_ds.remove_columns('fold')
  ds = ds.remove_columns('fold')
  
  # collate_fn=gezi.DictPadCollate()
  sampler = lele.get_sampler(ds, shuffle=True)
  kwargs = {'num_workers': FLAGS.num_workers, 'pin_memory': True, 'persistent_workers': True, 'collate_fn':collate_fn} 
  dl = torch.utils.data.DataLoader(ds, batch_size=gezi.batch_size(), sampler=sampler, **kwargs)
  
  valid_sampler = lele.get_sampler(valid_ds, shuffle=False)
  valid_dl = torch.utils.data.DataLoader(valid_ds,
                                         batch_size=gezi.eval_batch_size(),
                                         sampler=valid_sampler,
                                         **kwargs)
  
  model = AutoModelForMaskedLM.from_pretrained(FLAGS.backbone)
  model.resize_token_embeddings(len(tokenizer)) 
  fit(model,  
      dataset=dl,
      valid_dataset=valid_dl,
      opt_params=lele.get_opt_params(model, weight_decay=FLAGS.weight_decay),
    ) 
  
  if rank == 0:          
    tokenizer.save_pretrained(FLAGS.model_dir)
    model.save_pretrained(FLAGS.model_dir)
    logger.info('tokenizer and model config saved')
    
if __name__ == '__main__':
  app.run(main)  
