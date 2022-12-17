#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   model.py
#        \author   chenghuige
#          \date   2022-05-11 11:12:57.220407
#   \Description
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gezi.common import *

from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
try:
  from lele.nlp.pretrain.masklm import MaskLM
except Exception:
  pass

from src.config import *
from src.preprocess import *


class WeightedLayerPooling(nn.Module):

  def __init__(self,
               num_hidden_layers,
               layer_start=4,
               layer_weights=None):
    super(WeightedLayerPooling, self).__init__()
    self.layer_start = layer_start
    self.num_hidden_layers = num_hidden_layers
    self.layer_weights = layer_weights if layer_weights is not None \
        else nn.Parameter(
            torch.tensor([1] * (num_hidden_layers+1 - layer_start), dtype=torch.float)
        )

  def forward(self, all_hidden_states):
    all_layer_embedding = all_hidden_states[self.layer_start:, :, :, :]
    weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(
        -1).expand(all_layer_embedding.size())
    weighted_average = (weight_factor * all_layer_embedding).sum(
        dim=0) / self.layer_weights.sum()
    return weighted_average

def re_initializing_layer(model, config, layer_num):
  for module in model.encoder.layer[-layer_num:].modules():
    if isinstance(module, nn.Linear):
      module.weight.data.normal_(mean=0.0, std=config.initializer_range)
      if module.bias is not None:
        module.bias.data.zero_()
      elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=config.initializer_range)
        if module.padding_idx is not None:
          module.weight.data[module.padding_idx].zero_()
      elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
  return model

class Model(nn.Module):

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

    self.backbone, self.tokenizer = self.init_backbone(FLAGS.backbone)

    config = self.backbone.config
    dim = config.hidden_size
    
    # if FLAGS.concat_last:
    #   dim += config.hidden_size * (FLAGS.last_layers - 1)
    if FLAGS.concat_last:
      # self.pooler = WeightedLayerPooling(
      #   config.num_hidden_layers, 
      #   layer_start=config.num_hidden_layers - FLAGS.last_layers, 
      #   layer_weights=None)
      self.poolers = nn.ModuleList([WeightedLayerPooling(
        config.num_hidden_layers, 
        layer_start=config.num_hidden_layers - FLAGS.last_layers, 
        layer_weights=None) for _ in range(NUM_TARGETS + 1)])

    if FLAGS.encode_targets:
      if FLAGS.concat_cls:
        dim += config.hidden_size
        # if not FLAGS.concat_last:
        #   dim += config.hidden_size
        # else:
        #   dim += FLAGS.last_layers * config.hidden_size

    self.fc, self.fcs = None, None
    Linear = nn.Linear if not FLAGS.mdrop else lele.layers.MultiDropout
    if FLAGS.train_version > 0:
      self.fc = Linear(dim, 1)
    else:
      if FLAGS.encode_targets:
        num_targets = NUM_TARGETS
        # num_targets += 2
        if FLAGS.poolings:
          self.poolings = nn.ModuleList([
              lele.layers.Pooling(FLAGS.pooling, dim)
              for _ in range(NUM_TARGETS)
          ])
          dim += config.hidden_size
        ic(dim)
        if FLAGS.seq_encoder:
          dim_ = dim
          dim = int(dim / 2)
          RNN = getattr(nn, FLAGS.rnn_type)
          if not FLAGS.rnn_bi:
            self.seq_encoder = RNN(dim,
                                   dim,
                                   FLAGS.rnn_layers,
                                   dropout=FLAGS.rnn_dropout,
                                   bidirectional=False,
                                   batch_first=True)
          else:
            self.seq_encoder = RNN(dim,
                                   int(dim / 2),
                                   FLAGS.rnn_layers,
                                   dropout=FLAGS.rnn_dropout,
                                   bidirectional=True,
                                   batch_first=True)
          dim = dim_

        if not FLAGS.method == 'cls':
          if not FLAGS.fcs:
            self.fc = Linear(dim, 1)
          else:
            self.fcs = nn.ModuleList(
                [Linear(dim, 1) for _ in range(num_targets)])
        else:
          self.fc = Linear(dim, NUM_LABELS)
      else:
        if not FLAGS.poolings:
          self.pooling = lele.layers.Pooling(FLAGS.pooling, dim)
          dim = self.pooling.output_dim
          odim = NUM_TARGETS
          if not FLAGS.fcs:
            self.fc = Linear(dim, odim)
          else:
            if not FLAGS.mlp:
              self.fcs = nn.ModuleList(
                  [Linear(dim, 1) for _ in range(NUM_TARGETS)])
            else:
              if FLAGS.mlp_method == 1:
                self.fcs = nn.ModuleList([
                    lele.layers.MLP(dim, [512, 1]) for _ in range(NUM_TARGETS)
                ])
              else:
                self.fcs = nn.ModuleList([
                    lele.layers.MLP(dim, [512], 1) for _ in range(NUM_TARGETS)
                ])
        else:
          self.poolings = nn.ModuleList([
              lele.layers.Pooling(FLAGS.pooling, dim)
              for _ in range(NUM_TARGETS)
          ])
          if not FLAGS.fcs:
            self.fc = Linear(dim, 1)
          else:
            self.fcs = nn.ModuleList(
                [Linear(dim, 1) for _ in range(NUM_TARGETS)])

    if FLAGS.init_fc:
      if self.fc is not None:
        self.init_fc_(self.fc)
      if self.fcs is not None:
        for i in range(len(self.fcs)):
          self.init_fc_(self.fcs[i])

    if FLAGS.keras_init:
      for key, m in self.named_children():
        if key not in ['backbone']:
          lele.keras_init(m)

    if FLAGS.dropout > 0:
      self.dropout = nn.Dropout(FLAGS.dropout)

    if FLAGS.lm_train:
      self.lm = MaskLM(tokenizer=self.tokenizer)
      config.vocab_size = len(self.tokenizer)
      self.lm_header = BertOnlyMLMHead(config)

    if FLAGS.opt_8bit:
      # lele.set_embedding_parameters_bits(self.backbone.embeddings)
      pass
    elif FLAGS.opt_fused:
      lele.replace_with_fused_layernorm(self)

  def init_fc_(self, module):
    if isinstance(module, nn.Linear):
      module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
      if module.bias is not None:
        module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
      module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
      if module.padding_idx is not None:
        module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
      module.bias.data.zero_()
      module.weight.data.fill_(1.0)
      
  def init_backbone(self, backbone_name, model_dir=None, load_weights=False):
    backbone_dir = f'{os.path.dirname(FLAGS.model_dir)}/{FLAGS.backbone_dir}' if FLAGS.backbone_dir is not None else None
    model_dir = model_dir or backbone_dir or FLAGS.model_dir

    try:
      config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    except Exception as e:
      # logger.warning(e)
      try:
        config = AutoConfig.from_pretrained(backbone_name,
                                            trust_remote_code=True)
      except Exception:
        config = AutoConfig.from_pretrained(backbone_name.lower(),
                                            trust_remote_code=True)

    if FLAGS.type_vocab_size:
      config.update({'type_vocab_size': FLAGS.type_vocab_size})

    if FLAGS.max_position_embeddings:
      config.update(
          {'max_position_embeddings': FLAGS.test_maxlen or FLAGS.max_len})

    if FLAGS.concat_last:
      config.update({'output_hidden_states': True})

    if FLAGS.no_dropout:
      config.update({
          'attention_probs_dropout_prob': 0.,
          'hidden_dropout_prob': 0.,
      })

    self.config = config
    ic(self.config)

    ic(model_dir, backbone_name, os.path.exists(f'{model_dir}/model.pt'),
       os.path.exists(f'{model_dir}/pytorch_model.bin'))

    if FLAGS.lm_train and (not FLAGS.lm_header):
      backbone = AutoModelForMaskedLM.from_pretrained(
          FLAGS.backbone, trust_remote_code=True, ignore_mismatched_sizes=True)
    else:
      if os.path.exists(f'{model_dir}/model.pt') and (
          not os.path.exists(f'{model_dir}/pytorch_model.bin')):
        try:
          backbone = AutoModel.from_config(config, trust_remote_code=True)
          logger.info(f'backbone init from config')
        except Exception as e:
          # logger.warning(e)
          backbone = AutoModel.from_pretrained(backbone_name,
                                               config=config,
                                               trust_remote_code=True,
                                               ignore_mismatched_sizes=True)
          logger.info(f'backbone init from {backbone_name}')
      else:
        try:
          backbone = AutoModel.from_pretrained(model_dir,
                                               config=config,
                                               trust_remote_code=True,
                                               ignore_mismatched_sizes=True)
          logger.info(f'backbone init from {model_dir}')
        except Exception as e:
          # logger.warning(e)
          backbone = AutoModel.from_pretrained(backbone_name,
                                               config=config,
                                               trust_remote_code=True,
                                               ignore_mismatched_sizes=True)
          logger.info(f'backbone init from {backbone_name}')

    if os.path.exists(f'{model_dir}/config.json'):
      try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
      except Exception:
        tokenizer = get_tokenizer(backbone_name)
    else:
      tokenizer = get_tokenizer(backbone_name)

    backbone.resize_token_embeddings(len(tokenizer))
    logger.info('backbone resize done')
    if FLAGS.unk_init:
      try:
        # TODO bart not ok
        unk_id = tokenizer.unk_token_id
        with torch.no_grad():
          word_embeddings = lele.get_word_embeddings(backbone)
          br_id = tokenizer.convert_tokens_to_ids(BR)
          word_embeddings.weight[br_id, :] = word_embeddings.weight[unk_id, :]
      except Exception as e:
        ic(e)

    if FLAGS.freeze_emb:
      lele.freeze(lele.get_word_embeddings(backbone))

    if FLAGS.gradient_checkpointing:
      backbone.gradient_checkpointing_enable()
      
    if FLAGS.reinit_layers > 0:
      re_initializing_layer(backbone, config, FLAGS.reinit_layers)
    
    logger.info('backbone and tokenizer init done')
    return backbone, tokenizer

  def encode(self, inputs, idx=None):
    backbone, tokenizer = self.backbone, self.tokenizer

    m = {
        'input_ids': inputs['input_ids'],
    }

    if not FLAGS.disable_attention_mask:
      if 'attention_mask' in inputs:
        m['attention_mask'] = inputs['attention_mask']

    if not FLAGS.disable_token_type_ids:
      if 'token_type_ids' in inputs:
        m['token_type_ids'] = inputs['token_type_ids']

    x = backbone(**m)[0] if not FLAGS.concat_last else backbone(
        **m)['hidden_states']
    return x

  def fake_pred(self, inputs, requires_grad=False):
    input_ids = inputs['input_ids'] if not 0 in inputs else inputs[0][
        'input_ids']
    bs = input_ids.shape[0]
    return torch.rand([bs, 1],
                      device=input_ids.device,
                      requires_grad=requires_grad)

  def forward(self, inputs):
    if FLAGS.fake_infer:
      return {
          'pred': self.fake_pred(inputs, requires_grad=self.training),
      }

    res = {}

    if FLAGS.lm_train:
      input_ids, lm_label = self.lm.mask_tokens(inputs['input_ids'])
      res['label'] = lm_label
      inputs['input_ids'] = input_ids
      m = {
          'input_ids': input_ids,
          'attention_mask': inputs['attention_mask'],
      }
      if 'token_type_ids' in inputs:
        m['token_type_ids'] = inputs['token_type_ids']
      res['pred'] = self.lm_header(self.encode(m))
      return res

    x = self.encode(inputs)

    if FLAGS.train_version > 0:
      x = x[:, 0]
      res['pred'] = self.fc(x)
      return res

    if FLAGS.encode_targets:
      num_targets = NUM_TARGETS
      # num_targets += 2

      if FLAGS.concat_last:
        # l = [a[:, :1 + NUM_TARGETS] for a in x]
        # x = torch.cat(l[-FLAGS.last_layers:], -1)
        x = torch.stack(x)
        xs = [pooler(x) for pooler in self.poolers]
        # x = self.pooler(x)
        x = torch.stack([a[:, i] for i, a in enumerate(xs)], 1)

      if FLAGS.seq_encoder:
        x, _ = self.seq_encoder(x)

      if FLAGS.concat_cls:
        x_cls = x[:, :1].repeat(1, num_targets, 1)

      if FLAGS.poolings:
        xs = []
        for i, pooling in enumerate(self.poolings):
          xs.append(pooling(x))
        x_pooling = torch.stack(xs, 1)
      x = x[:, 1:1 + num_targets]
      if FLAGS.concat_cls:
        x = torch.cat([x, x_cls], -1)
      if FLAGS.poolings:
        x = torch.cat([x, x_pooling], -1)
      if not FLAGS.method == 'cls':
        if not FLAGS.fcs:
          x = self.fc(x)
        else:
          x = torch.cat([self.fcs[i](x[:, i]) for i in range(num_targets)], -1)
        res['pred'] = x.squeeze(-1)
      else:
        res['pred'] = self.fc(x)
    else:
      if not FLAGS.poolings:
        if not FLAGS.concat_last:
          x = self.pooling(x)
        else:
          # x = torch.cat([x[-1][:, 0], x[-2][:, 0], x[-3][:, 0], x[-4][:, 0]],
          #               -1)
          pass
        if not FLAGS.fcs:
          res['pred'] = self.fc(x)
        else:
          res['pred'] = torch.cat([self.fcs[i](x) for i in range(NUM_TARGETS)],
                                  -1)
      else:
        xs = []
        for i, pooling in enumerate(self.poolings):
          if not FLAGS.fcs:
            xs.append(self.fc(pooling(x)))
          else:
            xs.append(self.fcs[i](pooling(x)))
        res['pred'] = torch.cat(xs, -1)

    # ic(res['pred'].shape)

    if not 'pred' in res:
      res['pred'] = self.fake_pred(inputs)
    return res

  def get_loss_fn(self):
    from src.torch.loss import calc_loss
    return calc_loss
