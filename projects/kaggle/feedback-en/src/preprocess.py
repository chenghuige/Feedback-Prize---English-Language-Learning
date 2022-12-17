#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   preprocess.py
#        \author   chenghuige
#          \date   2022-05-11 11:12:36.045278
#   \Description
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gezi.common import *
from src.config import *
from src.util import *

# https://www.kaggle.com/code/abhishek/multi-label-stratified-folds
def create_mlabel_folds(data, num_splits):
  data["fold"] = -1
  from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
  mskf = MultilabelStratifiedKFold(n_splits=num_splits, shuffle=True, random_state=FLAGS.fold_seed)
  labels = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]
  data_labels = data[labels].values

  for f, (t_, v_) in enumerate(mskf.split(data, data_labels)):
    data.loc[v_, "fold"] = f

  return data


def set_fold(df):
  assert 'text_id' in df.columns
  gezi.set_fold_worker(df, FLAGS.folds, 80, seed=FLAGS.fold_seed)
  if FLAGS.mlabel_cv:
    df = create_mlabel_folds(df, FLAGS.folds)
  
def clear():
  dfgs.clear()

# 1, 1.5,   2,  2.5, 3,  3.5, 4, 4.5, 5
# 0 0.125 0.25 0.375 0.5 
# 0,1,2,3,4,5,6,7,8
def convert_label(label):
  if FLAGS.method == 'cls':
    label = (label - 1) / 0.5
  else:
    if FLAGS.norm_target:
      label = (label - 1) * 0.25 
    # if (FLAGS.target is not None) and (not FLAGS.nontarget_weight):
    #   mask = np.zeros(NUM_TARGETS)
    #   # mask[FLAGS.target] = 1.
    #   mask[2], mask[3], mask[4], mask[5] = 1, 1, 1, 1
    #   label = label * mask + (1 - mask) * -100
  return label

# tool = None
def prepare(df, mark):
  if FLAGS.use_v1:
    if 'id' in df.columns:
      df['text_id'] = df['id']

  if 'score' in df.columns:
    df['label'] = df['score']
    gezi.set_fold_worker(df, FLAGS.folds, 80, seed=FLAGS.fold_seed)
  else:
    if TARGETS[0] in df.columns:
      df['label'] = list(np.stack([df[target].values for target in TARGETS], 1))
      df['label'] = df.label.apply(convert_label)
      set_fold(df)
    else:
      ic(FLAGS.use_v1, FLAGS.mode)
      if (not FLAGS.use_v1) or (FLAGS.mode == 'test'):
        df['label'] = [[-1] * len(TARGETS)] * len(df)
        gezi.set_worker(df, 80)
        df['fold'] = -1
      else:
        df['label'] = 0
        set_fold(df)
  
  if mark == 'pseudo_train':
    # df['label'] = [[-1] * len(TARGETS)] * len(df)
    df['fold'] = -1
    
  if FLAGS.resolve_encodings:
    df['full_text'] = df.full_text.apply(lambda x: resolve_encodings_and_normalize(x))
  df['n_words'] = df.full_text.apply(lambda x: len(x.split()))
  # if FLAGS.grammer_tool:
  #   global tool
  # if tool is None:
  #   import language_tool_python
  #   tool = language_tool_python.LanguageTool('en-US')
  #   df['grammer_error'] = df.full_text.apply(lambda x: len(tool.check(x)) / len(x))
  if FLAGS.readability:
    import readability
    df['full_text'] = df.full_text.apply(lambda x: str(int(readability.getmeasures(x, lang='en')['readability grades']['ARI'] * 1000)) + ' ' + x)
  if FLAGS.convert_br:
    df['full_text'] = df.full_text.apply(lambda x: x.replace('\n', BR)) 
  df['id'] = df['text_id']
  tokenizer = get_tokenizer(FLAGS.backbone)
  df['input_ids'] = gezi.prun_list(tokenizer.encode, df.full_text.values, FLAGS.workers, desc='tokenize')
  df['n_tokens'] = df.input_ids.apply(len)
  
  if mark == 'test':
    df = df.sort_values('n_tokens', ascending=False)
          
  return df

dfs = {}
def get_df(mark='train'):
  ic(dfs.keys())    
  if mark in dfs:
    return dfs[mark]
  
  ifile = f'{FLAGS.root}/train.csv'
  if FLAGS.train_version:
    ifile = f'{FLAGS.root}/train_v{FLAGS.train_version}.csv'
  ifile_ = ifile
  if mark == 'test' and (not FLAGS.train_version):
    ifile = f'{FLAGS.root}/test.csv'
  if FLAGS.use_v1:
    ifile = f'{FLAGS.root}/train_v1.csv'
  if mark == 'pseudo_train':
    if FLAGS.online:
      ifile = f'{FLAGS.root}/pseudo_train.csv'
    else:
      ifile = f'{FLAGS.root}/pseudo_train_{FLAGS.fold}.csv'
  
  if mark != 'test':
    if ifile.endswith(f'/{mark}.csv'):
      if os.path.exists(f'../working/{mark}.fea'):
        ifile = f'../working/{mark}.fea'
  ic(mark, ifile)
  if ifile.endswith('.csv'):
    df = pd.read_csv(ifile)
  else:
    df = pd.read_feather(ifile)
  
  hack_infer = False
  if FLAGS.hack_infer or FLAGS.simulate_test:
    if len(df) < 100:
      df = pd.read_csv(ifile_)
      hack_infer = True
      mark = 'train'
   
  # train_1 also treat as test mode since no label
  df = prepare(df, mark=mark)  
  if hack_infer:
    if not FLAGS.simulate_test:
      df = df[df.fold==FLAGS.fold]
    text_ids = gezi.unique_list(df.text_id.values)
    ic(len(text_ids))
    if not FLAGS.simulate_test:
      text_ids = text_ids[:int(len(text_ids) * FLAGS.hack_rate)]
    else:
      text_ids = text_ids[:2700]

    ic(len(text_ids))
    text_ids = set(text_ids)
    df = df[df.text_id.isin(text_ids)]
    
  dfs[mark] = df  
  
  if FLAGS.trans:
    if not FLAGS.trans in dfs:
      if FLAGS.trans != 'mix':
        df_ = pd.read_csv(f'{FLAGS.root}/train_{FLAGS.trans}.csv')
        df_['full_text'] = df_[f'trans_{FLAGS.trans}']
        df_ = prepare(df_, mark=mark)  
        dfs[FLAGS.trans] = df_
      else:
        df_ = pd.read_csv(f'{FLAGS.root}/train_nl.csv')
        df_['full_text'] = df_[f'trans_nl']
        df_ = prepare(df_, mark=mark)  
        df2_ = pd.read_csv(f'{FLAGS.root}/train_fr.csv')
        df2_['full_text'] = df2_[f'trans_fr']
        df2_ = prepare(df2_, mark=mark)  
        # df_ = pd.concat([df_, df2_]).reset_index()
        df_ = pd.concat([df, df_, df2_]).reset_index()
        dfs[FLAGS.trans] = df_
    
  return df

tokenizers = {}
def get_tokenizer(backbone):
  if backbone in tokenizers:
    return tokenizers[backbone]
  
  from transformers import AutoTokenizer
  if 'cocolm' in backbone:
    from cocolm.tokenization_cocolm import COCOLMTokenizer as AutoTokenizer
  try:
    tokenizer = AutoTokenizer.from_pretrained(FLAGS.model_dir, use_fast=True)
  except Exception:
    tokenizer = AutoTokenizer.from_pretrained(backbone, use_fast=True)

  if tokenizer.convert_tokens_to_ids(BR) == tokenizer.unk_token_id:
    assert len(BR) == 1
    tokenizer.add_tokens([BR], special_tokens=False)
      
  tokenizers[backbone] = tokenizer
  return tokenizer

def encode(text, max_len, last_tokens=0, trunct=True, padding='longest'):
  if 'cocolm' in FLAGS.backbone:
    text = text.replace('[SEP]', '。')
  tokenizer = get_tokenizer(FLAGS.backbone)
  res = {}
  input_ids = tokenizer.encode(text)
  if FLAGS.test_maxlen:
    assert len(input_ids) <= FLAGS.test_maxlen
  if trunct:
    input_ids = gezi.trunct(input_ids, max_len, last_tokens=last_tokens + 1)
  attention_mask = [1] * len(input_ids)
  res = {
    'input_ids': input_ids,
    'attention_mask': attention_mask
  }
  if 'cocolm' in FLAGS.backbone:
    for i in range(len(res['input_ids'])):
      if res['input_ids'][i] == tokenizer.convert_tokens_to_ids('。'):
        res['input_ids'][i] = tokenizer.sep_token_id
  return res

def parse_example(row, tokenizer=None, subset=None, rng=None):
  fe = row.copy()
  if 'label' not in fe:
    fe['label'] = [-1] * len(TARGETS)
    
  if FLAGS.mask_label:
    l = list(range(NUM_TARGETS))
    rng.shuffle(l)
    fe['weights'] = [1.] * NUM_TARGETS
    fe['weights'][l[0]] = 0.1
    fe['weights'][l[1]] = 0.2
    fe['weights'][l[2]] = 0.3
    fe['weights'][l[3]] = 0.4
    fe['weights'][l[4]] = 0.5
    fe['weights'][l[5]] = 1.
    
  padding = 'max_length' if FLAGS.static_inputs_len else 'longest'
  text = row['full_text']
  sep = ' ' if not FLAGS.text_sep else '[SEP]'
  # if FLAGS.grammer_tool:
  #   text = str(int(row['grammer_error'] * 1000)) + sep + text
  if FLAGS.encode_targets:
    targets = TARGETS
    # targets = ['min', 'max', *targets]
    text = ' '.join(targets) + sep + text 
    # n_words = fe['n_words']
    # n_tokens = fe['n_tokens']
    # wt_ratio = int((n_words / n_tokens) * 1000)
    # text = ' '.join(targets) + sep + f'{n_words} {n_tokens} {wt_ratio}' + sep + text 
  max_len = FLAGS.test_maxlen or FLAGS.max_len
  fe.update(encode(text, max_len, FLAGS.last_tokens, padding))
  return fe

def get_datasets(valid=True, mark='train'):
  from datasets import Dataset
  df = get_df(mark)

  tokenizer = get_tokenizer(FLAGS.backbone)
  # ic(tokenizer)

  ds = Dataset.from_pandas(df)

  # num_proc = cpu_count() if FLAGS.pymp else 1
  num_proc = 4 if FLAGS.pymp else 1
  # gezi.try_mkdir(f'{FLAGS.root}/cache')
  records_name = get_records_name()
  
  ds = ds.map(
      lambda example: parse_example(example, tokenizer=tokenizer),
      remove_columns=ds.column_names,
      batched=False,
      num_proc=num_proc,
      # cache_file_name=f'{FLAGS.root}/cache/{records_name}.infer{int(infer)}.arrow' if not infer else None
  )
  
  ic(ds)
  
  ignore_feats = [key for key in ds.features if ds.features[key].dtype == 'string' or (ds.features[key].dtype == 'list' and ds.features[key].feature.dtype == 'string')]
  ic(ignore_feats)
  
  infer = mark == 'test'
  if infer:
    m = {}
    for key in ignore_feats:
      m[key] = ds[key]
    gezi.set('infer_dict', m)
    ds = ds.remove_columns(ignore_feats)
    return ds

  if not FLAGS.online:
    train_ds = ds.filter(lambda x: x['fold'] != FLAGS.fold, num_proc=num_proc)
  else:
    train_ds = ds
  eval_ds = ds.filter(lambda x: x['fold'] == FLAGS.fold, num_proc=num_proc)

  m = {}
  for key in ignore_feats:
    m[key] = eval_ds[key]
  gezi.set('eval_dict', m)

  # also ok if not remove here
  train_ds = train_ds.remove_columns(ignore_feats)
  eval_ds = eval_ds.remove_columns(ignore_feats)
  ic(train_ds, eval_ds)
  if valid:
    valid_ds = ds.filter(lambda x: x['fold'] == FLAGS.fold, num_proc=num_proc)
    valid_ds = valid_ds.remove_columns(ignore_feats)
    return train_ds, eval_ds, valid_ds
  else:
    return train_ds, eval_ds

def get_dataloaders(valid=True, test_only=False):
  from src.torch.dataset import Dataset
  # collate_fn = gezi.DictPadCollate()
  collate_fn = gezi.NpDictPadCollate()
  kwargs = {
      'num_workers': FLAGS.num_workers,
      'pin_memory': FLAGS.pin_memory,
      'persistent_workers': FLAGS.persistent_workers,
      'collate_fn': collate_fn,
  }
  if (FLAGS.use_v1 or FLAGS.train_version) and FLAGS.mode != 'test':
    test_dl = None
  else:
    test_ds = Dataset('test')
    ic(len(test_ds))
    sampler_test = lele.get_sampler(test_ds, shuffle=False)
    test_dl = torch.utils.data.DataLoader(test_ds,
                                          batch_size=gezi.eval_batch_size(),
                                          sampler=sampler_test,
                                          **kwargs)
    if test_only:
      return test_dl
  
  train_ds = Dataset('train')
  eval_ds = Dataset('valid')
  if (not FLAGS.online) and (not FLAGS.use_v1):
    assert len(set(train_ds.df.id) & set(eval_ds.df.id)) == 0
    assert len(set(train_ds.df.text_id) & set(eval_ds.df.text_id)) == 0
  if valid:
    valid_ds = Dataset('valid')
  
  # if valid:
  #   train_ds, eval_ds, valid_ds = get_datasets(valid=True)
  # else:
  #   train_ds, eval_ds = get_datasets(valid=False)
  
  sampler = lele.get_sampler(train_ds, shuffle=True)
  # melt.batch_size 全局总batch大小，FLAGS.batch_size 单个gpu的batch大小，gezi.batch_size做batch的时候考虑兼容distributed情况下的batch_size
  train_dl = torch.utils.data.DataLoader(train_ds,
                                         batch_size=gezi.batch_size(),
                                         sampler=sampler,
                                         drop_last=FLAGS.drop_last,
                                         **kwargs)
  sampler_eval = lele.get_sampler(eval_ds, shuffle=False)
  eval_dl = torch.utils.data.DataLoader(eval_ds,
                                        batch_size=gezi.eval_batch_size(),
                                        sampler=sampler_eval,
                                        **kwargs)
  if valid:
    sampler_valid = lele.get_sampler(valid_ds, shuffle=False)
    valid_dl = torch.utils.data.DataLoader(valid_ds,
                                           batch_size=gezi.eval_batch_size(),
                                           sampler=sampler_valid,
                                           **kwargs)
    return train_dl, eval_dl, valid_dl, test_dl
  else:
    return train_dl, eval_dl, test_dl


def get_tf_datasets(valid=True):
  if valid:
    train_ds, eval_ds, valid_ds = get_datasets(valid=True)
  else:
    train_ds, eval_ds = get_datasets(valid=False)
  collate_fn = gezi.DictPadCollate(return_tensors='tf')
  train_ds = train_ds.to_tf_dataset(
      columns=train_ds.columns,
      label_cols=["label"],
      shuffle=True,
      collate_fn=collate_fn,
      batch_size=gezi.batch_size(),
  )
  eval_ds = eval_ds.to_tf_dataset(
      columns=eval_ds.columns,
      label_cols=["label"],
      shuffle=False,
      collate_fn=collate_fn,
      batch_size=gezi.eval_batch_size(),
  )
  if valid:
    valid_ds = valid_ds.to_tf_dataset(
        columns=valid_ds.columns,
        label_cols=["label"],
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=gezi.eval_batch_size(),
    )
    return train_ds, eval_ds, valid_ds
  else:
    return train_ds, eval_ds


def create_datasets(valid=True):
  if FLAGS.torch:
    return get_dataloaders(valid)
  else:
    return get_tf_datasets(valid)
  