#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   main.py
#        \author   chenghuige  
#          \date   2022-05-11 11:12:05.887273
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
sys.path.append('..')
sys.path.append('../../../../utils')
sys.path.append('../../../../third')
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["NCCL_DEBUG"] = 'WARNING'

from gezi.common import *

import tensorflow as tf
from tensorflow import keras
import torch
from torch.utils.data import DataLoader

import src
import src.eval as ev
from src import config
from src.config import *
from src import util
from src.preprocess import *
from src import postprocess

def main(_):
  fit = mt.fit  
  
  rank = gezi.get('RANK', 0)
  if rank != 0:
    ic.disable()
    
  config.init()
  mt.init()
  config.show()
  
  if FLAGS.ensemble_metrics:
    args = ' '.join(sys.argv[1:])
    gezi.system(f'./ensemble.py {args} --mn={FLAGS.mn}')
    exit(0)
  
  # NOTICE will exit after prepare so if DDP will fail for other process but that's fine
  if rank == 0:
    os.system(f'cp -rf ../src {FLAGS.model_dir}')
    os.system(f'cp ../dataset-metadata.json {FLAGS.model_dir}')
    if FLAGS.kaggle_prepare or os.path.exists(f'{FLAGS.model_dir}/done.txt'):
      gezi.prepare_kaggle_dataset(f'{MODEL_NAME}-model')
  
  if not FLAGS.torch:
    from src.tf.model import Model
  else:
    from src.torch.model import Model
  
  strategy = mt.distributed.get_strategy()
  with strategy.scope():    
    model = Model()
    model.eval_keys = [
                       'text_id', 'label', 'n_words', 
                      ]
    model.str_keys = ['text_id', 'full_text'] if FLAGS.tf_dataset else []
    model.out_keys = [] 
    
    callbacks = util.get_callbacks(model)
    if FLAGS.torch:     
      if not FLAGS.llrd:
        opt_params = lele.get_opt_params(model, backbone_lr=FLAGS.lr, 
                                         base_lr=FLAGS.base_lr, weight_decay=FLAGS.weight_decay,
                                         backbone=model.backbone)
      else:
        opt_params = util.llrd(model)
        # ic(opt_params)
      gezi.set('opt_params', opt_params)
    
    # out_hook = None
    # if FLAGS.torch:
    #   out_hook = postprocess.out_hook 
    # train from tfrecords input pre gen
    if FLAGS.tf_dataset:
      from src.dataset import Dataset
      fit(model,  
          Dataset=Dataset,
          # out_hook=out_hook,
          eval_fn=ev.evaluate,
          valid_write_fn=ev.valid_write,
          infer_write_fn=ev.infer_write,
          callbacks=callbacks
          ) 
    else:
      train_ds, eval_ds, valid_ds, test_ds = create_datasets()
      fit(model,  
          dataset=train_ds,
          eval_dataset=eval_ds,
          valid_dataset=valid_ds,
          test_dataset=test_ds,
          # out_hook=out_hook,
          eval_fn=ev.evaluate,
          valid_write_fn=ev.valid_write,
          infer_write_fn=ev.infer_write,
          callbacks=callbacks
          ) 
  
  if rank == 0:    
    if FLAGS.work_mode != 'test':
      if not FLAGS.online:
        gezi.folds_metrics(FLAGS.model_dir, FLAGS.folds)
    
    gezi.save(gezi.get('eval:x'), f'{FLAGS.model_dir}/valid.pkl', verbose=True)
    
    if FLAGS.work_mode == 'train':
      if FLAGS.online:
        if not FLAGS.torch:
          model.build_model().save_weights(f'{FLAGS.model_dir}/model2.h5')  
      gezi.save_model(model, FLAGS.model_dir, fp16=True)
      logger.info('model saved')
      model.tokenizer.save_pretrained(FLAGS.model_dir)
      model.config.save_pretrained(FLAGS.model_dir)
      logger.info('tokenizer and model config saved')
      
    if FLAGS.online:
      gezi.prepare_kaggle_dataset(f'{MODEL_NAME}-model')
    
if __name__ == '__main__':
  app.run(main)  
