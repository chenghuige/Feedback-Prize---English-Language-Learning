#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   mlm.py
#        \author   chenghuige  
#          \date   2022-04-28 09:05:19.995996
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gezi
from gezi.common import * 

import torch
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers import AutoTokenizer

"""
Modified from HuggingFace/transformers (https://github.com/huggingface/transformers/blob/0a3d0e02c5af20bfe9091038c4fd11fb79175546/src/transformers/data/data_collator.py#L102). 
It is a little bit faster cuz 
- intead of a[b] a on gpu b on cpu, tensors here are all in the same device
- don't iterate the tensor when create special tokens mask
And
- doesn't require huggingface tokenizer
- cost you only 550 µs for a (128,128) tensor on gpu, so dynamic masking is cheap   
"""
def mask_tokens(inputs, mask_token_index, vocab_size, special_token_indices, mlm_probability=0.15, replace_prob=0.1, orginal_prob=0.1, ignore_index=-100):
  """ 
  Prepare masked tokens inputs/labels for masked language modeling: (1-replace_prob-orginal_prob)% MASK, replace_prob% random, orginal_prob% original within mlm_probability% of tokens in the sentence. 
  * ignore_index in nn.CrossEntropy is default to -100, so you don't need to specify ignore_index in loss
  """
  
  device = inputs.device
  labels = inputs.clone()
  
  # Get positions to apply mlm (mask/replace/not changed). (mlm_probability)
  probability_matrix = torch.full(labels.shape, mlm_probability, device=device)
  special_tokens_mask = torch.full(inputs.shape, False, dtype=torch.bool, device=device)
  for sp_id in special_token_indices:
    special_tokens_mask = special_tokens_mask | (inputs==sp_id)
  probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
  mlm_mask = torch.bernoulli(probability_matrix).bool()
  labels[~mlm_mask] = ignore_index  # We only compute loss on mlm applied tokens

  # mask  (mlm_probability * (1-replace_prob-orginal_prob))
  mask_prob = 1 - replace_prob - orginal_prob
  mask_token_mask = torch.bernoulli(torch.full(labels.shape, mask_prob, device=device)).bool() & mlm_mask
  inputs[mask_token_mask] = mask_token_index

  # replace with a random token (mlm_probability * replace_prob)
  if int(replace_prob)!=0:
    rep_prob = replace_prob/(replace_prob + orginal_prob)
    replace_token_mask = torch.bernoulli(torch.full(labels.shape, rep_prob, device=device)).bool() & mlm_mask & ~mask_token_mask
    random_words = torch.randint(vocab_size, labels.shape, dtype=torch.long, device=device)
    inputs[replace_token_mask] = random_words[replace_token_mask]

  # do nothing (mlm_probability * orginal_prob)
  pass

  return inputs, labels, mlm_mask

# https://github.com/zr2021/2021_QQ_AIAC_Tack1_1st/blob/main/job1/data/masklm.py
class MaskLM(object):

  def __init__(self, tokenizer, mlm_probability=0.15):
    self.mlm_probability = mlm_probability
    self.tokenizer = tokenizer

  # def mask_tokens(self,
  #                 inputs: Any,
  #                 special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
  #   """
  #       Prepare masked tokens inputs/label for masked language modeling: 80% MASK, 10% random, 10% original.
  #   """
  #   inputs = inputs.long()
  #   device = inputs.device
  #   label = inputs.clone()
  #   # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
  #   probability_matrix = torch.full(label.shape,
  #                                   self.mlm_probability,
  #                                   device=device)
  #   if special_tokens_mask is None:
  #     special_tokens_mask = [
  #         self.tokenizer.get_special_tokens_mask(
  #             val, already_has_special_tokens=True) for val in label.tolist()
  #     ]
  #     special_tokens_mask = torch.tensor(special_tokens_mask,
  #                                        dtype=torch.bool,
  #                                        device=device)
  #   else:
  #     special_tokens_mask = special_tokens_mask.bool()

  #   probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
  #   masked_indices = torch.bernoulli(probability_matrix).bool()
  #   label[~masked_indices] = -100  # We only compute loss on masked tokens

  #   # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
  #   indices_replaced = torch.bernoulli(
  #       torch.full(label.shape, 0.8, device=device)).bool() & masked_indices
  #   inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
  #       self.tokenizer.mask_token)

  #   # 10% of the time, we replace masked input tokens with random word
  #   indices_random = torch.bernoulli(
  #       torch.full(label.shape, 0.5,
  #                  device=device)).bool() & masked_indices & ~indices_replaced
  #   random_words = torch.randint(len(self.tokenizer),
  #                                label.shape,
  #                                dtype=torch.long,
  #                                device=device)
  #   inputs[indices_random] = random_words[indices_random]

  #   # The rest of the time (10% of the time) we keep the masked input tokens unchanged
  #   return inputs, label

  def mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """
    import torch

    labels = inputs.clone()
    device = inputs.device
    # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
    probability_matrix = torch.full(labels.shape, self.mlm_probability, device=device)
    if special_tokens_mask is None:
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool, device=device)
    else:
        special_tokens_mask = special_tokens_mask.bool()

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8, device=device)).bool() & masked_indices
    inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5, device=device)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long, device=device)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels

class MaskVision(object):

  def __init__(self, mlm_probability=0.15):
    self.mlm_probability = mlm_probability

  def mask_frames(self, vision_feature, vision_mask):
    device = vision_feature.device
    probability_matrix = torch.full(vision_mask.shape,
                                    0.9 * self.mlm_probability,
                                    device=device)
    probability_matrix = probability_matrix * vision_mask

    masked_indices = torch.bernoulli(probability_matrix).bool()

    vision_label_index = torch.arange(
        vision_feature.size(0) * vision_feature.size(1),
        device=device).view(-1, vision_feature.size(1))
    vision_label_index = -100 * ~masked_indices + vision_label_index * masked_indices

    # 90% mask vision fill all 0.0
    masked_indices_unsqueeze = masked_indices.unsqueeze(-1).expand_as(
        vision_feature)
    inputs = vision_feature.data.masked_fill(masked_indices_unsqueeze, 0.0)
    label = vision_feature[masked_indices_unsqueeze].contiguous().view(
        -1, vision_feature.size(2))

    return inputs, vision_label_index


class Shufflevision(object):

  def __init__(self):
    pass

  def torch_shuf_vision(self, vision_feature):
    bs = vision_feature.size()[0]
    # batch 内前一半 vision 保持原顺序，后一半 vision 逆序
    shuf_index = torch.tensor(
        list(range(bs // 2)) + list(range(bs // 2, bs))[::-1])
    # shuf 后的 label
    label = (torch.tensor(list(range(bs))) == shuf_index).float()
    vision_feature = vision_feature[shuf_index]
    return vision_feature, label


def calc_mfm_loss(vision_feature_output,
                  vision_feature_input,
                  vision_mask,
                  vision_label_index,
                  normalize=False,
                  temp=0.1):
  if normalize:
    vision_feature_output = torch.nn.functional.normalize(vision_feature_output,
                                                          p=2,
                                                          dim=2)
    vision_feature_input = torch.nn.functional.normalize(vision_feature_input,
                                                         p=2,
                                                         dim=2)

  afm_scores_tr = vision_feature_output.view(-1,
                                             vision_feature_output.shape[-1])

  vision_tr = vision_feature_input.permute(2, 0, 1)
  vision_tr = vision_tr.view(vision_tr.shape[0], -1)

  logits_matrix = torch.mm(afm_scores_tr, vision_tr)
  if normalize:
    logits_matrix = logits_matrix / temp

  vision_mask_float = vision_mask.to(dtype=torch.float)
  mask_matrix = torch.mm(vision_mask_float.view(-1, 1),
                         vision_mask_float.view(1, -1))
  masked_logits = logits_matrix + (1. - mask_matrix) * -1e8

  logpt = F.log_softmax(masked_logits, dim=-1)
  logpt = torch.diag(logpt)
  nce_loss = -logpt

  vision_label_index_mask = (vision_label_index != -100)
  nce_loss = nce_loss.masked_select(vision_label_index_mask.view(-1))
  nce_loss = nce_loss.mean()
  return nce_loss
