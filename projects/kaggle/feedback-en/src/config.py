#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   config.py
#        \author   chenghuige  
#          \date   2022-05-11 11:18:18.068436
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gezi.common import * 

RUN_VERSION = '16'
SUFFIX = ''
MODEL_NAME = 'feedbacken'

flags.DEFINE_integer('fold_seed', 42, '')
flags.DEFINE_alias('fs', 'fold_seed')
flags.DEFINE_bool('tf', False, '')
flags.DEFINE_string('root', '../input/feedback-prize-english-language-learning', '')
flags.DEFINE_string('root_v2', '../input/feedback-prize-effectiveness', '')
flags.DEFINE_string('root_v1', '../input/feedback-prize-2021', '')
flags.DEFINE_bool('use_v1', False, '')
flags.DEFINE_bool('add_v1', False, '')
flags.DEFINE_bool('v1_only', False, '')
flags.DEFINE_bool('v1_ft', False, '')
flags.DEFINE_bool('v1_dynamic', False, '')
flags.DEFINE_bool('pre_dynamic', False, '')
flags.DEFINE_string('trans_ft', None, '')
flags.DEFINE_bool('exclude_v2', True, '')
flags.DEFINE_integer('train_version', 0, '')
flags.DEFINE_alias('train_ver', 'train_version')
flags.DEFINE_bool('v2_ft', False, '')

flags.DEFINE_bool('norm_target', False, '')
flags.DEFINE_bool('mlabel_cv', True, '')
flags.DEFINE_bool('aux_loss', False, '')
flags.DEFINE_float('aux_loss_rate', 0., '')
flags.DEFINE_bool('aux_abs', False, '')
flags.DEFINE_bool('grammer_tool', False, '')
flags.DEFINE_bool('use_weights', False, '')
flags.DEFINE_bool('init_fc', False, '')
flags.DEFINE_bool('keras_init', False, '')
flags.DEFINE_bool('max_loss', False, '')
flags.DEFINE_bool('add_max_loss', False, '')
flags.DEFINE_bool('mask_label', False, '')
flags.DEFINE_integer('reinit_layers', 0, '')
flags.DEFINE_bool('llrd', False, '')
flags.DEFINE_bool('readability', False, '')

flags.DEFINE_string('method', 'reg', 'reg:regression cls:classification')

flags.DEFINE_string('trans', None, '')
flags.DEFINE_integer('trans_method', 0, '')

flags.DEFINE_bool('read_essay', False, '')
flags.DEFINE_string('hug', 'deberta-v3', '')
flags.DEFINE_string('backbone', None, '')
flags.DEFINE_string('backbone_dir', None, '')
flags.DEFINE_bool('lower', False, '')
flags.DEFINE_integer('max_labels', 30, '')
flags.DEFINE_integer('max_len', None, '')
flags.DEFINE_integer('last_tokens', 0, '')
flags.DEFINE_bool('static_inputs_len', False, '')
flags.DEFINE_bool('token_types', False, '')
flags.DEFINE_bool('convert_br', True, '')
flags.DEFINE_bool('unk_init', True, '')
flags.DEFINE_bool('mask_init', False, '')
flags.DEFINE_bool('mean_init', False, '')
flags.DEFINE_bool('mask_discourse', False, '')
flags.DEFINE_bool('encodes', True, '')
flags.DEFINE_integer('encodes_method', 2, '')
flags.DEFINE_bool('listwise', False, '')
flags.DEFINE_bool('listwise_cls', False, '')
flags.DEFINE_bool('listwise_pooling', False, '')
flags.DEFINE_integer('listwise_strategy', 0, '')
flags.DEFINE_bool('disable_token_type_ids', False, '')
flags.DEFINE_alias('disable_ttids', 'disable_token_type_ids')
flags.DEFINE_bool('disable_attention_mask', False, '')
flags.DEFINE_bool('first_token_only', False, '')
flags.DEFINE_bool('token_cls', False, '')
flags.DEFINE_string('pooling_mask_key', 'token_type_ids', '')
flags.DEFINE_string('segment_key', 'token_type_ids', '')
flags.DEFINE_float('dropout', 0., '')
flags.DEFINE_bool('no_dropout', False, '')
flags.DEFINE_string('loss_reduction', 'mean', '')
flags.DEFINE_bool('labels_loss', False, '')
flags.DEFINE_bool('soft_label', False, '')
flags.DEFINE_string('loss_fn', 'mse', '')
flags.DEFINE_float('focal_alpha', 1, '')
flags.DEFINE_float('focal_gamma', 2, '')
flags.DEFINE_float('base_loss_rate', 1., '')
flags.DEFINE_float('rank_loss_rate', 0., '')
flags.DEFINE_string('rank_loss', 'pearson', '')
flags.DEFINE_bool('resolve_encodings', True, '')

flags.DEFINE_integer('test_maxlen', None, '')

flags.DEFINE_bool('encode_start', False, '')
flags.DEFINE_bool('encode_start_ratio', False, '')
flags.DEFINE_integer('encode_spans', 100, '')
flags.DEFINE_bool('dynamic_context', False, '')

flags.DEFINE_bool('max_position_embeddings', False, '')
flags.DEFINE_integer('type_vocab_size', None, '')
flags.DEFINE_bool('position_biased_input', None, '')

flags.DEFINE_bool('multi_lr', False, '')
flags.DEFINE_bool('continue_pretrain', False, '')
flags.DEFINE_float('base_lr', None, '1e-3 or 5e-4')
flags.DEFINE_bool('weight_decay', True, '')
flags.DEFINE_bool('mdrop', False, '')

flags.DEFINE_bool('freeze_emb', False, '')
flags.DEFINE_string('pooling', 'latt', 'latt better then cls, att offline same, online seems a bit better but 5 folds still latt a bit better')
flags.DEFINE_bool('pooling_mask', True, '')
flags.DEFINE_bool('poolings', False, '')
flags.DEFINE_bool('fcs', False, '')
flags.DEFINE_bool('text_sep', False, '')

flags.DEFINE_bool('encode_targets', False, '')
flags.DEFINE_integer('target', None, '')
flags.DEFINE_float('nontarget_weight', 0, '')
flags.DEFINE_float('target_weight', 10, '')
flags.DEFINE_bool('concat_cls', False, '')
flags.DEFINE_bool('concat_last', False, '')
flags.DEFINE_integer('last_layers', 4, '')
flags.DEFINE_bool('mlp', False, '')
flags.DEFINE_integer('mlp_method', 1, '')

flags.DEFINE_bool('seq_encoder', False, 'seq encoder helps')
flags.DEFINE_integer('rnn_layers', 1, '')
flags.DEFINE_bool('rnn_bi', True, '')
flags.DEFINE_float('rnn_dropout', 0.1, '')
flags.DEFINE_string('rnn_type', 'LSTM', '')
flags.DEFINE_bool('rnn_double_dim', False, '')

flags.DEFINE_float('aug_prob', 0., '')
flags.DEFINE_float('mask_prob', 0.15, '')
flags.DEFINE_integer('aug_seed', None, '')
flags.DEFINE_bool('mask_self', True, '')

flags.DEFINE_bool('layer_norm', False, '')

flags.DEFINE_bool('use_context', True, '')

flags.DEFINE_bool('ensemble_select', False, '')

flags.DEFINE_string('custom_backbone', None, '')
flags.DEFINE_bool('lm_use_type', True, '')
flags.DEFINE_bool('lm_train', False, 'mlm now, TODO rtd then deberta v3')
flags.DEFINE_integer('lm_max_len', 512, '')
flags.DEFINE_integer('lm_stride', 128, '')
flags.DEFINE_bool('lm_header', False, 'custom lm header bad valid result and train loss higher at begging, because it is not BertOnlyHeader.. TODO')
flags.DEFINE_bool('lm_line_by_line', True, '')
flags.DEFINE_bool('dynamic_line_by_line', True, '')
flags.DEFINE_integer('lm_max_lines', 0, '')
flags.DEFINE_bool('lm_custom_dataset', False, '')
flags.DEFINE_bool('lm_balance', True, '')
flags.DEFINE_bool('lm_hug_inputs', False, '')

#Unicode编号	 U+02B6
BR = 'ʶ'

hugs = {
  'longformer': 'allenai/longformer-large-4096',  # ok 但是注意tf版本的longformer启动特别慢，虽然运行比torch稍微快一些, tf训练也更加占用显存 不能4A100训练bs16 torch可以
  'large': 'allenai/longformer-large-4096',  
  'longformer-base': 'allenai/longformer-base-4096',  # ok
  'base': 'allenai/longformer-base-4096',  
  'tiny': 'roberta-base',
  'roberta': 'roberta-large', # ok
  'mid': 'roberta-large',
  'electra': 'google/electra-large-discriminator', # ok electra版本效果不错 甚至好于roberta512 但是某些fold有很大概率loss不下降崩溃 没找到原因 但是online全量训练应该还好可以看train自己loss是否正常 在线提交融合结果也还ok
  'electra-base': 'google/electra-base-discriminator',
  'bart': 'facebook/bart-large', # okk
  'bart-base': 'facebook/bart-base',
  'large-qa': 'allenai/longformer-large-4096-finetuned-triviaqa',
  'roberta-base': 'roberta-base',
  'ro': 'roberta-base',
  'fast': 'roberta-base',
  'roberta-large': 'roberta-large',
  'bird': 'google/bigbird-roberta-base', # ok but not good
  'bigbird': 'google/bigbird-roberta-base',
  'bird-large': 'google/bigbird-roberta-large',
  'pegasus': 'google/pegasus-large',
  'scico': 'allenai/longformer-scico',
  'xlnet-large': 'xlnet-large-cased',
  'xlnet': 'xlnet-large-cased',
  'robertam': 'roberta-large-mnli',
  'robertas': 'deepset/roberta-large-squad2',
  'roberta-squad': 'deepset/roberta-large-squad2',
  'reformer': 'google/reformer-enwik8', #fail
  'roformer': 'junnyu/roformer_chinese_base',
  'span': 'SpanBERT/spanbert-large-cased', # need --br='[SEP]'
  'gpt2': 'gpt2-large', # not well
  'berts': 'phiyodr/bart-large-finetuned-squad2',
  'barts': 'phiyodr/bart-large-finetuned-squad2',
  'bart-squad': 'phiyodr/bart-large-finetuned-squad2', # this is file
  'albert': 'albert-large-v2',
  'bert-cased': 'bert-large-cased',
  'bert-uncased': 'bert-large-uncased',
  'bert-squad': 'deepset/bert-large-uncased-whole-word-masking-squad2', # fail
  't5': 't5-large',
  'base-squad': 'valhalla/longformer-base-4096-finetuned-squadv1',
  'albert-squad': 'mfeb/albert-xxlarge-v2-squad2',
  'electra-squad': 'ahotrod/electra_large_discriminator_squad2_512',
  'deberta': 'microsoft/deberta-large',
  'deberta-base': 'microsoft/deberta-base',
  'deberta-xl': 'microsoft/deberta-xlarge',
  'deberta-xlarge': 'microsoft/deberta-xlarge',
  'deberta-v2': 'microsoft/deberta-v2-xlarge', # v2 v3 all has problem... of tokenizer no fast/non python version (hack now)
  'deberta-v2-xlarge': 'microsoft/deberta-v2-xlarge', # v2 v3 all has problem... of tokenizer no fast/non python version (hack now)
  'deberta-v2-xxlarge': 'microsoft/deberta-v2-xxlarge',
  'deberta-v3': 'microsoft/deberta-v3-large', 
  'deberta-v3-large': 'microsoft/deberta-v3-large', 
  'deberta-v3-large-squad2': 'deepset/deberta-v3-large-squad2',
  'deberta-v3-base': 'microsoft/deberta-v3-base',
  'deberta-v3-small': 'microsoft/deberta-v3-small', 
  'patent': 'anferico/bert-for-patents',
  'patent-cpc': 'bradgrimm/patent-cpc-predictor',
  'bart-patent': 'Pyke/bart-finetuned-with-patent',
  'peasus-patent': 'google/pegasus-big_patent',
  'coco': 'microsoft/cocolm-large',
  'psbert': 'AI-Growth-Lab/PatentSBERTa',
  'simcse-patent': 'Yanhao/simcse-bert-for-patent',
  'xlm-roberta-large': 'xlm-roberta-large',
  'xlm': 'xlm-roberta-large',
  'unilm': 'microsoft/unilm-large-cased',
  'rembert': 'google/rembert',
  'pubmed': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
  'pubmed-full': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
  'biobert': 'emilyalsentzer/Bio_ClinicalBERT',
  'bern2-ner': 'dmis-lab/bern2-ner',
  'biobert-large': 'dmis-lab/biobert-large-cased-v1.1-squad',
  'bert-base-uncased_clinical-ner': 'samrawal/bert-base-uncased_clinical-ner',
  'xlm-roberta': 'llange/xlm-roberta-large-english-clinical',
  'funnel': 'funnel-transformer/large',
  'funnel-m': 'funnel-transformer/medium',
  'funnel-i': 'funnel-transformer/intermediate',
  'erine': 'nghuyong/ernie-2.0-large-en',
  'mpnet': 'microsoft/mpnet-base',
  'fnet-base': 'google/fnet-base',
  'deberta-spell': 'stuartmesham/deberta-v3-large_lemon-spell_5k_1_p3',
  'all-mpnet-base-v2': 'sentence-transformers/all-mpnet-base-v2',
}

cat2label = {
  'Ineffective': 0,
  'Adequate': 1,
  'Effective': 2,
}

label2cat = {v: k for k, v in cat2label.items()}

NUM_CLASSES = len(cat2label)

classes = [
  'Claim', 'Evidence', 'Position', 'Concluding Statement', 'Lead', 'Counterclaim', 'Rebuttal'
]

id2dis =  {
  0: 'Nothing',
  1: 'Claim',
  2: 'Evidence',
  3: 'Position',
  4: 'Concluding Statement',
  5: 'Lead',
  6: 'Counterclaim',
  7: 'Rebuttal'
 }

dis2id = {
  k: v for v, k in id2dis.items()
}

# 衔接 句法 词汇 用语 语法 惯例
TARGETS = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
NUM_TARGETS = len(TARGETS)

LABELS = np.asarray([1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
NUM_LABELS = len(LABELS)

WEIGHTS = [1, 1, 1, 1, 1, 1]
def get_weights(reshape=True):
  weights = np.asarray(WEIGHTS)
  if reshape:
    weights = weights.reshape(1, -1)
  return weights

def get_backbone(backbone, hug):
  backbone = backbone or hugs.get(hug, hug)
  backbone_ = backbone.split('/')[-1]
  if FLAGS.continue_pretrain:
    if FLAGS.custom_backbone:
      backbone_ = FLAGS.custom_backbone
    backbone_path = f'{FLAGS.root}/pretrain/{backbone_}'
    if os.path.exists(f'{backbone_path}/config.json'):
      backbone = backbone_path
      return backbone
  backbone_path = f'{FLAGS.root}/{backbone_}'
  if os.path.exists(f'{backbone_path}/config.json'):
    backbone = backbone_path
    return backbone
    
  return backbone

def get_records_name():
  records_name = FLAGS.backbone.split('/')[-1]
  return records_name

def config_train():
  # offline online show 2e-5 better for deberta-v3-large
  lr = 2e-5
  bs = 64
  # FLAGS.lr_decay_power = 2
  # FLAGS.lr_decay_power = 0.5
  # FLAGS.lr_decay_power = 2
  
  if FLAGS.torch and (not FLAGS.tf_dataset):
    FLAGS.drop_last = True
  
  # if 'xlarge' in FLAGS.hug:
  #   lr = 1e-5
  #   # bs = 64
  #   FLAGS.grad_acc *= 4
    
  # if 'electra' in FLAGS.hug:
  #   lr = 1e-5
    
  # if 'deberta-v2-xlarge' in FLAGS.hug:
  #   lr = 1e-5
  #   bs = 64
  #   # FLAGS.lr_decay_power = 2
  
  FLAGS.lr = FLAGS.lr or lr
  # FLAGS.clip_gradients = 1.
  
  # versy sensitive to lr ie, for roformer v2 large, 5e-5 + 5e-4 will not converge but 5e-5 + 1e-4 will
  # also TODO with 1e-4 + 1e-3 lr, opt_fused very interesting download with roformer v2 large layer norm .. random init due to key miss in checkpoint will converge
  # but if save_pretrained then reload will not, why ?
  if FLAGS.multi_lr:
    # base_lr = FLAGS.lr * 10.
    base_lr = 1e-3
    FLAGS.base_lr = FLAGS.base_lr or base_lr
  
  if FLAGS.grad_acc == 1:
    pass
  #   if FLAGS.max_len > 160:
  #     FLAGS.grad_acc *= 2
    
  #     if FLAGS.rdrop_rate > 0:
  #       FLAGS.grad_acc *= 2
        
  #   if FLAGS.max_len > 512:
  #     FLAGS.grad_acc *= 2
  #     if FLAGS.max_len > 768:
  #       FLAGS.grad_acc *= 2
  #       # if FLAGS.max_len > 1280:
  #       #   FLAGS.grad_acc *= 2
  elif FLAGS.grad_acc == 0:
    FLAGS.grad_acc = 1
  
  # FLAGS.loss_scale = 100
  FLAGS.bs = FLAGS.bs or bs
  FLAGS.bs = max(FLAGS.bs, gezi.get_num_gpus())
  FLAGS.eval_bs = FLAGS.eval_bs or FLAGS.bs / FLAGS.grad_acc * 4
  
  ep = 3 
  FLAGS.ep = FLAGS.ep or ep
  
  # change from adamw back to adam
  optimizer = 'adamw' 
  FLAGS.optimizer = FLAGS.optimizer or optimizer
  FLAGS.opt_eps = 1e-7
  
  # scheduler = 'cosine'
  # linear seems better for len512 rdrop_rate 1 online 603->597
  scheduler = 'linear' 
  FLAGS.scheduler = FLAGS.scheduler or scheduler

  if FLAGS.tf_dataset:
    records_pattern = f'{FLAGS.root}/{FLAGS.records_name}/{get_records_name()}/train/*.tfrec'
    ic(records_pattern)
    files = gezi.list_files(records_pattern) 
    FLAGS.valid_files = [x for x in files if int(os.path.basename(x).split('.')[0]) % FLAGS.folds == FLAGS.fold]
    if FLAGS.online:
      FLAGS.train_files = files
    else:
      FLAGS.train_files = [x for x in files if x not in FLAGS.valid_files]
    
    ic(FLAGS.train_files[:2], FLAGS.valid_files[:2])
    
    if not FLAGS.train_files:
      FLAGS.tf_dataset = False
  
def config_model():
  global WEIGHTS
  
  FLAGS.backbone = get_backbone(FLAGS.backbone, FLAGS.hug)
  
  if FLAGS.v2_ft:
    if not FLAGS.pretrain:
      if 'deberta-v3-large' in FLAGS.backbone:
        FLAGS.pretrain = '/work/pikachu/projects/kaggle/feedback-eff/working/online/10/0/deberta-v3.len1280.encode2.flag-listwise-s3.encode_type-0'
      elif 'deberta-v3-base' in FLAGS.backbone:
        FLAGS.pretrain = '/work/pikachu/projects/kaggle/feedback-eff/working/online/10/0/deberta-v3-base.len1280.encode2.flag-listwise-s3.encode_type-0'
      elif 'deberta-v3-small' in FLAGS.backbone:
        FLAGS.pretrain = '/work/pikachu/projects/kaggle/feedback-eff/working/online/10/0/deberta-v3-small.len1280.encode2.flag-listwise-3.encode_type-0'
        
        
  max_len = 1280
  FLAGS.max_len = FLAGS.max_len or max_len
 
  awp_train = True
  FLAGS.awp_train = awp_train if FLAGS.awp_train is None else FLAGS.awp_train
  fgm_train = False
  FLAGS.fgm_train = fgm_train if FLAGS.fgm_train is None else FLAGS.fgm_train
  # if not any(x in FLAGS.hug for x in ['xlarge', 'roberta', 'electra']):
  #   FLAGS.rdrop_rate = 0.1
  # FLAGS.rdrop_rate = 1 # tested rdrop make better 
    
  if any(x in FLAGS.hug for x in ['roberta', 'xlnet', 'patent']):
    FLAGS.find_unused_parameters = True
  
  # if FLAGS.max_len > 512 and 'deberta' in FLAGS.backbone:
  #   FLAGS.max_position_embeddings = FLAGS.max_len
  
  if 'logformer' in FLAGS.backbone:
    FLAGS.disable_token_type_ids = True
  # if 'logformer' in FLAGS.backbone:
  #   assert FLAGS.disable_token_type_ids or FLAGS.type_vocab_size
  
  if FLAGS.encodes_method == 4:
    FLAGS.type_vocab_size = FLAGS.max_labels
    FLAGS.labels_loss = True
  
  if FLAGS.listwise:
    if FLAGS.segment_key == 'token_type_ids':
      FLAGS.type_vocab_size = FLAGS.max_labels + 1
    else:
      FLAGS.type_vocab_size = len(dis2id) + 1

  if FLAGS.no_dropout:
    FLAGS.dropout = 0.
    # FLAGS.rnn_dropout = 0.
    # FLAGS.rdrop_rate = 0.
    
  if FLAGS.target is not None:
    FLAGS.loss_reduction = 'none'
    
    if FLAGS.nontarget_weight:
      WEIGHTS = [FLAGS.nontarget_weight] * NUM_TARGETS
      WEIGHTS[FLAGS.target] = 1.
    
    if FLAGS.target_weight:
      WEIGHTS = [1.] * NUM_TARGETS
      WEIGHTS[FLAGS.target] = FLAGS.target_weight
  
  if FLAGS.trans:
    # if FLAGS.trans in ['es', 'cn']:
    #   FLAGS.trans_method = 1
    # if FLAGS.trans in ['nl']:
      # FLAGS.trans_method = 2
    FLAGS.trans_method = 1
    FLAGS.ep = 2
    FLAGS.awp_train = False
    # ft fr, de
    if FLAGS.trans_method > 0:
      # ft4 es, cn
      FLAGS.use_weights = True
      # set grammer as 0
      WEIGHTS[-2] = 0.
      if FLAGS.trans_method > 1:
        # ft3 nl
        # set convention as 0
        WEIGHTS[-1] = 0.
        
  if FLAGS.v1_only:
    FLAGS.ep = 1
    FLAGS.nvs = 4
    
  if FLAGS.trans and FLAGS.add_v1:
    FLAGS.ep = 1
    FLAGS.nvs = 4
  
  if FLAGS.use_weights:
    FLAGS.loss_reduction = 'none'   
  
  if FLAGS.train_version == 1:
    FLAGS.encode_targets = False   

  if FLAGS.max_loss:
    FLAGS.loss_reduction = 'none'  
    
  if FLAGS.mask_label:
    FLAGS.loss_reduction = 'none'  
  
  if FLAGS.backbone in ['deberta-large']:
    FLAGS.max_len = min(FLAGS.max_len, 1024)

  if FLAGS.hug.startswith('fnet'):
    FLAGS.disable_attention_mask = True
  
def show():
  ic(
     FLAGS.wandb,
     FLAGS.wandb_entity,
     FLAGS.wandb_group,
     FLAGS.backbone,
     FLAGS.max_len, 
     FLAGS.encode_targets,
     FLAGS.concat_cls,
     FLAGS.concat_last,
     FLAGS.disable_token_type_ids,
     FLAGS.dropout,
     FLAGS.lr,
     FLAGS.grad_acc,
     FLAGS.scheduler,
     FLAGS.bs,
     FLAGS.pooling,
     FLAGS.awp_train,
     FLAGS.adv_start_epoch,
     FLAGS.rdrop_rate,
     get_weights(),
     )
    
def init():
  config_model()

  folds = 5
  FLAGS.folds = FLAGS.folds or folds
  FLAGS.fold = FLAGS.fold or 0
  # FLAGS.show_keys = ['score']

  FLAGS.buffer_size = 20000
  FLAGS.static_input = True
  FLAGS.cache_valid = True
  FLAGS.async_eval = True
  FLAGS.async_eval_last = True if not FLAGS.pymp else False
  FLAGS.async_valid = False
  
  # FLAGS.find_unused_parameters=True
  
  if not FLAGS.tf:
    # make torch by default
    FLAGS.torch = True
  else:
    FLAGS.torch = False
  
  ic(FLAGS.torch, FLAGS.torch_only, FLAGS.tf_dataset)

  
  FLAGS.run_version = FLAGS.run_version or RUN_VERSION
      
  if FLAGS.online:
    FLAGS.allow_train_valid = True
    FLAGS.nvs = 1
    # assert FLAGS.fold == 0
    # if FLAGS.fold != 0:
    #   ic(FLAGS.fold)
    #   exit(0)
     
  if FLAGS.log_all_folds or FLAGS.fold == 0:
    wandb = True
    if FLAGS.wandb is None:
      FLAGS.wandb = wandb
    FLAGS.wandb_project = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # FLAGS.wandb_entity = 'pikachu'
  FLAGS.write_summary = True
  
  FLAGS.run_version += f'/{FLAGS.fold}'
  
  pres = ['offline', 'online']
  pre = pres[int(FLAGS.online)]
  model_dir = f'../working/{pre}/{FLAGS.run_version}/model'  
  FLAGS.model_dir = FLAGS.model_dir or model_dir
  if FLAGS.mn == 'model':
    FLAGS.mn = ''
  if not FLAGS.mn:  
    if FLAGS.hug:
      model_name = FLAGS.hug
    FLAGS.mn = model_name   
    FLAGS.mn += f'.len{FLAGS.max_len}'
    if FLAGS.tf:
      FLAGS.mn = f'tf.{FLAGS.mn}' 
    ignores = ['tf', 'hug', 'test_file', 'static_inputs_len', 'max_len', 'encodes_method']
    # if FLAGS.mode == 'test':
    #   ignores += ['use_v1']
    mt.model_name_from_args(ignores=ignores)
    FLAGS.mn += SUFFIX 

    if FLAGS.v1_ft:
      if not FLAGS.pretrain:
        if not FLAGS.trans_ft:
          if 'v1_ft' in FLAGS.mn:
            FLAGS.pretrain = FLAGS.mn.replace('v1_ft', 'v1_only')
          else:
            FLAGS.pretrain = FLAGS.mn + '.v1_only'
        else:
          if 'v1_ft' in FLAGS.mn:
            FLAGS.pretrain = FLAGS.mn.replace('v1_ft', 'add_v1')
          else:
            FLAGS.pretrain = FLAGS.mn + '.add_v1'
          
        if FLAGS.v1_dynamic:
          if '.flag-encode_targets-cls.' in FLAGS.pretrain:
            FLAGS.pretrain = FLAGS.pretrain.replace('.flag-encode_targets-cls.', '.flag-encode_targets-cls-crank.')
          else:
            FLAGS.pretrain = FLAGS.pretrain.replace('.flag-encode_targets-cls-crank.', '.flag-encode_targets-cls.')
          FLAGS.pretrain = FLAGS.pretrain.replace('.v1_dynamic', '')
      
      if FLAGS.trans_ft:
        FLAGS.pretrain = FLAGS.pretrain.replace('.trans_ft', '.trans')
              
      ic(FLAGS.pretrain)
      assert FLAGS.pretrain
  else:  
    if FLAGS.trans_ft:
      FLAGS.pretrain = FLAGS.mn.replace('.trans_ft', '.trans')
    if FLAGS.pre_dynamic:
      if '.flag-encode_targets-cls.' in FLAGS.pretrain:
        FLAGS.pretrain = FLAGS.pretrain.replace('.flag-encode_targets-cls.', '.flag-encode_targets-cls-crank.')
      else:
        FLAGS.pretrain = FLAGS.pretrain.replace('.flag-encode_targets-cls-crank.', '.flag-encode_targets-cls.')
      FLAGS.pretrain = FLAGS.pretrain.replace('.pre_dynamic', '')
  
  config_train()
  
  if not FLAGS.online:
    # if FLAGS.ep == 1:
    #   FLAGS.nvs = FLAGS.nvs or 10
    # else:
    FLAGS.nvs = FLAGS.nvs or FLAGS.ep
    # FLAGS.vie = 1
    
  FLAGS.write_valid_final = True
  FLAGS.save_model = False
  FLAGS.sie = 1e10  
  # sie = 1
  # if FLAGS.sie is None:
  #   FLAGS.sie = sie
  
  if FLAGS.lm_train:
    FLAGS.seq_encoder = False
    FLAGS.sie = 1
    FLAGS.do_valid = False
    FLAGS.do_test = False
    FLAGS.awp_train = False
    FLAGS.rdrop_rate = 0.
