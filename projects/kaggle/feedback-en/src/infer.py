#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   infer.py
#        \author   chenghuige  
#          \date   2022-05-15 07:00:15.332073
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
if os.path.exists('/kaggle'):
  sys.path.append('/kaggle/input/pikachu/utils')
  sys.path.append('/kaggle/input/pikachu/third')
  sys.path.append('.')
else:
  sys.path.append('..')
  sys.path.append('../../../../utils')
  sys.path.append('../../../../third')

from gezi.common import *
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["NCCL_DEBUG"] = 'WARNING'

from src.config import *
from src.preprocess import *
from src.postprocess import *

flags.DEFINE_string('ofile', '../working/x.pkl', '')
flags.DEFINE_bool('infer_norm', False, '')

def main(argv):
  ic.enable()
  ic('infer.py start')
  model_dir = FLAGS.model_dir
  bs = FLAGS.eval_bs
  out_file = FLAGS.ofile
  infer_norm = FLAGS.infer_norm
  ic(bs)
  gezi.restore_configs(model_dir, ignores=gezi.get_commandline_flags())
  ic(model_dir, bs, FLAGS.eval_bs, FLAGS.bs)
  show()
  FLAGS.train_allnew = False
  FLAGS.grad_acc = 1
  FLAGS.restore_configs = False
  FLAGS.bs, FLAGS.eval_bs = bs, bs
  mt.init()
  FLAGS.model_dir = model_dir
  FLAGS.pymp = False
  FLAGS.num_workers = 2
  FLAGS.pin_memory = True
  FLAGS.persistent_workers = True
  FLAGS.workers = 1
  FLAGS.fold = 0
  FLAGS.backbone_dir = None
  if FLAGS.listwise:
    FLAGS.max_len = 2560
       
  if gezi.in_kaggle():
    backbone = FLAGS.backbone.split('/')[-1].replace('_', '-')
    FLAGS.backbone = '../input/' + backbone
  
  FLAGS.mode = 'test'
  FLAGS.work_mode = 'test'
  FLAGS.model_dir = model_dir
  ic(FLAGS.backbone, FLAGS.model_dir, os.path.exists(f'{FLAGS.model_dir}/model.pt'))
  try:
    display(pd.read_csv(f'{model_dir}/metrics.csv'))
  except Exception as e:
    logger.warning(e)
  
  test_ds = get_dataloaders(test_only=True)
  
  if not FLAGS.torch:
    from src.tf.model import Model
  else:
    from src.torch.model import Model
  model = Model()
  
  logger.info('before load weights')
  ic(gezi.get_mem_gb())
  gezi.load_weights(model, model_dir)
  ic(gezi.get_mem_gb())
  logger.info('load weights done')

  logger.info('before predict')
  x = lele.predict(model, test_ds, out_keys=['id', 'label', 'discourse_ids', 'discourse_types', 'n_essay_words'], 
                   amp=FLAGS.amp_infer, fp16=FLAGS.fp16_infer)
  logger.info('predict done')
  if FLAGS.listwise:
    x = transforms(x)
  
  ic(infer_norm)
  if infer_norm:
    x['pred'] = to_pred(x['pred'])
  #m = gezi.get('infer_dict')
  #x['id'] = m['id']
  ic(x)
  gezi.save(x, out_file)
  
  gc.collect()
  torch.cuda.empty_cache()

if __name__ == '__main__':
  app.run(main)  
  
