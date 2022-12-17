#!/usr/bin/env python3
# Copyright 2018-present, HKUST-KnowComp.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Definitions of model layers/NN modules"""

from os import access
import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
from torch.autograd import Variable
import math
import random

from absl import flags

FLAGS = flags.FLAGS

import gezi

logging = gezi.logging

import lele

# ------------------------------------------------------------------------------
# Modules
# ------------------------------------------------------------------------------


class CudnnRnn(nn.Module):
  """Stacked Bi-directional RNNs.

    Differs from standard PyTorch library in that it has the option to save
    and concat the hidden states between layers. (i.e. the output hidden size
    for each sequence input is num_layers * hidden_size).
    for hkust tf rent, reccruent dropout, bw drop out, concat layers and rnn padding
    WELL TOO SLOW 0.3hour/per epoch -> 1.5 hour..
    """

  def __init__(self,
               input_size,
               hidden_size,
               num_layers,
               dropout_rate=0,
               dropout_output=False,
               rnn_type=nn.LSTM,
               concat_layers=False,
               recurrent_dropout=True,
               bw_dropout=False,
               padding=False):
    super(CudnnRnn, self).__init__()
    self.padding = padding
    self.dropout_output = dropout_output
    self.dropout_rate = dropout_rate
    self.num_layers = num_layers
    self.concat_layers = concat_layers
    self.recurrent_dropout = recurrent_dropout
    self.bw_dropout = bw_dropout
    self.fws = nn.ModuleList()
    self.bws = nn.ModuleList()

    if type(rnn_type) == str:
      rnn_type = nn.GRU if rnn_type == 'gru' else nn.LSTM

    self.num_units = []
    for i in range(num_layers):
      input_size = input_size if i == 0 else 2 * hidden_size
      self.num_units.append(input_size)
      self.fws.append(
          rnn_type(input_size, hidden_size, num_layers=1, bidirectional=False))
      self.bws.append(
          rnn_type(input_size, hidden_size, num_layers=1, bidirectional=False))

  def forward(self, x, x_mask, fw_masks=None, bw_masks=None):
    """Encode either padded or non-padded sequences.

        Can choose to either handle or ignore variable length sequences.
        Always handle padding in eval.

        Args:
            x: batch * len * hdim
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            x_encoded: batch * len * hdim_encoded
        """
    # Transpose batch and sequence dims
    batch_size = x.size(0)
    x = x.transpose(0, 1)

    length = (1 - x_mask).sum(1)

    # Encode all layers
    outputs = [x]
    for i in range(self.num_layers):
      rnn_input = outputs[-1]
      mask = None
      if self.dropout_rate > 0:
        if not self.recurrent_dropout:
          rnn_input_ = F.dropout(rnn_input,
                                 p=self.dropout_rate,
                                 training=self.training)
        else:
          if fw_masks is None:
            mask = F.dropout(torch.ones(1, batch_size,
                                        self.num_units[i]).cuda(),
                             p=self.dropout_rate,
                             training=self.training)
          else:
            mask = fw_masks[i]
          rnn_input_ = rnn_input * mask

      # Forward
      fw_output = self.fws[i](rnn_input_)[0]

      mask = None
      if self.dropout_rate > 0 and self.bw_dropout:
        if not self.recurrent_dropout:
          rnn_input_ = F.dropout(rnn_input,
                                 p=self.dropout_rate,
                                 training=self.training)
        else:
          if bw_masks is None:
            mask = F.dropout(torch.ones(1, batch_size,
                                        self.num_units[i]).cuda(),
                             p=self.dropout_rate,
                             training=self.training)
          else:
            mask = bw_masks[i]
          rnn_input_ = rnn_input * mask

      rnn_input_ = lele.reverse_padded_sequence(rnn_input_, length)

      bw_output = self.bws[i](rnn_input_)[0]

      bw_output = lele.reverse_padded_sequence(bw_output, length)

      outputs.append(torch.cat([fw_output, bw_output], 2))

    # Concat hidden layers
    if self.concat_layers:
      output = torch.cat(outputs[1:], 2)
    else:
      output = outputs[-1]

    # Transpose back
    output = output.transpose(0, 1)

    # Dropout on output layer
    if self.dropout_output and self.dropout_rate > 0:
      output = F.dropout(output, p=self.dropout_rate, training=self.training)
    return output.contiguous()


# TODO support recurrent dropout! to be similar as tf version, support shared dropout
class StackedBRNN(nn.Module):
  """Stacked Bi-directional RNNs.

    Differs from standard PyTorch library in that it has the option to save
    and concat the hidden states between layers. (i.e. the output hidden size
    for each sequence input is num_layers * hidden_size).
    """

  def __init__(self,
               input_size,
               hidden_size,
               num_layers,
               dropout_rate=0,
               recurrent_dropout=False,
               dropout_output=False,
               rnn_type=nn.LSTM,
               concat_layers=False,
               padding=False):
    super(StackedBRNN, self).__init__()
    self.padding = padding
    self.dropout_output = dropout_output
    self.dropout_rate = dropout_rate
    self.num_layers = num_layers
    self.concat_layers = concat_layers
    self.rnns = nn.ModuleList()
    self.recurrent_dropout = recurrent_dropout
    if type(rnn_type) is str:
      rnn_type = nn.GRU if rnn_type == 'gru' else nn.LSTM
    self.num_units = []
    for i in range(num_layers):
      input_size = input_size if i == 0 else 2 * hidden_size
      self.num_units.append(input_size)
      self.rnns.append(
          rnn_type(input_size, hidden_size, num_layers=1, bidirectional=True))

  def forward(self, x, x_mask):
    """Encode either padded or non-padded sequences.

        Can choose to either handle or ignore variable length sequences.
        Always handle padding in eval.

        Args:
            x: batch * len * hdim
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            x_encoded: batch * len * hdim_encoded
        """
    if x_mask.data.sum() == 0:
      # No padding necessary.
      output = self._forward_unpadded(x, x_mask)
    elif self.padding or not self.training:
      # Pad if we care or if its during eval.
      output = self._forward_padded(x, x_mask)
    else:
      # We don't care.
      output = self._forward_unpadded(x, x_mask)

    return output.contiguous()

  def _forward_unpadded(self, x, x_mask):
    """Faster encoding that ignores any padding."""
    # Transpose batch and sequence dims
    x = x.transpose(0, 1)
    batch_size = x.size(1)

    # Encode all layers
    outputs = [x]
    for i in range(self.num_layers):
      rnn_input = outputs[-1]

      # Apply dropout to hidden input
      if self.dropout_rate > 0:
        if not self.recurrent_dropout:
          rnn_input = F.dropout(rnn_input,
                                p=self.dropout_rate,
                                training=self.training)
        else:
          mask = F.dropout(torch.ones(1, batch_size, self.num_units[i]).cuda(),
                           p=self.dropout_rate,
                           training=self.training)
          rnn_input = rnn_input * mask

      # Forward
      rnn_output = self.rnns[i](rnn_input)[0]
      outputs.append(rnn_output)

    # Concat hidden layers
    if self.concat_layers:
      output = torch.cat(outputs[1:], 2)
    else:
      output = outputs[-1]

    # Transpose back
    output = output.transpose(0, 1)

    # Dropout on output layer
    if self.dropout_output and self.dropout_rate > 0:
      output = F.dropout(output, p=self.dropout_rate, training=self.training)
    return output

  def _forward_padded(self, x, x_mask):
    """Slower (significantly), but more precise, encoding that handles
        padding.
        #chg I think do not need padded if you not want to use last state, last output
        """
    # Compute sorted sequence lengths
    #lengths = x_mask.data.eq(0).long().sum(1).squeeze()
    # TODO seems not need squeeze wich might cause error when batch_size is 1
    lengths = x_mask.data.eq(0).long().sum(1)
    _, idx_sort = torch.sort(lengths, dim=0, descending=True)
    _, idx_unsort = torch.sort(idx_sort, dim=0)

    lengths = list(lengths[idx_sort])
    idx_sort = Variable(idx_sort)
    idx_unsort = Variable(idx_unsort)

    # Sort x
    x = x.index_select(0, idx_sort)

    # Transpose batch and sequence dims
    x = x.transpose(0, 1)
    batch_size = x.size(1)

    # Pack it up
    rnn_input = nn.utils.rnn.pack_padded_sequence(x, lengths)

    # Encode all layers
    outputs = [rnn_input]
    for i in range(self.num_layers):
      rnn_input = outputs[-1]

      # Apply dropout to input
      if self.dropout_rate > 0:
        if not self.recurrent_dropout:
          dropout_input = F.dropout(rnn_input.data,
                                    p=self.dropout_rate,
                                    training=self.training)
        else:
          # TODO recurrent_dropout not work now, as mask (1, 2, 1024) but rnn_input.data should be like [881, 1024]
          mask = F.dropout(torch.ones(1, batch_size, self.num_units[i]).cuda(),
                           p=self.dropout_rate,
                           training=self.training)
          dropout_input = rnn_input.data * mask
        rnn_input = nn.utils.rnn.PackedSequence(dropout_input,
                                                rnn_input.batch_sizes)
      outputs.append(self.rnns[i](rnn_input)[0])

    # Unpack everything
    for i, o in enumerate(outputs[1:], 1):
      outputs[i] = nn.utils.rnn.pad_packed_sequence(o)[0]

    # Concat hidden layers or take final
    if self.concat_layers:
      output = torch.cat(outputs[1:], 2)
    else:
      output = outputs[-1]

    # Transpose and unsort
    output = output.transpose(0, 1)
    output = output.index_select(0, idx_unsort)

    # Pad up to original batch sequence length
    if output.size(1) != x_mask.size(1):
      padding = torch.zeros(output.size(0),
                            x_mask.size(1) - output.size(1),
                            output.size(2)).type(output.data.type())
      output = torch.cat([output, Variable(padding)], 1)

    # Dropout on output layer
    if self.dropout_output and self.dropout_rate > 0:
      output = F.dropout(output, p=self.dropout_rate, training=self.training)
    return output


class FeedForwardNetwork(nn.Module):

  def __init__(self, input_size, hidden_size, output_size, dropout_rate=0, activation='relu'):
    super(FeedForwardNetwork, self).__init__()
    self.dropout_rate = dropout_rate
    self.linear1 = nn.Linear(input_size, hidden_size)
    self.linear2 = nn.Linear(hidden_size, output_size)
    self.activation = getattr(F, activation)
  def forward(self, x):
    x_proj = F.dropout(self.activation(self.linear1(x)),
                       p=self.dropout_rate,
                       training=self.training)
    x_proj = self.linear2(x_proj)
    return x_proj


class PointerNetwork(nn.Module):

  def __init__(self,
               x_size,
               y_size,
               hidden_size,
               dropout_rate=0,
               cell_type=nn.GRUCell,
               normalize=True):
    super(PointerNetwork, self).__init__()
    self.normalize = normalize
    self.hidden_size = hidden_size
    self.dropout_rate = dropout_rate
    self.linear = nn.Linear(x_size + y_size, hidden_size, bias=False)
    self.weights = nn.Linear(hidden_size, 1, bias=False)
    self.self_attn = NonLinearSeqAttn(y_size, hidden_size)
    self.cell = cell_type(x_size, y_size)

  def init_hiddens(self, y, y_mask):
    attn = self.self_attn(y, y_mask)
    res = attn.unsqueeze(1).bmm(y).squeeze(1)  # [B, I]
    return res

  def pointer(self, x, state, x_mask):
    x_ = torch.cat([x, state.unsqueeze(1).repeat(1, x.size(1), 1)], 2)
    s0 = torch.tanh(self.linear(x_))
    s = self.weights(s0).view(x.size(0), x.size(1))
    s.data.masked_fill_(x_mask.data, -float('inf'))
    a = F.softmax(s)
    res = a.unsqueeze(1).bmm(x).squeeze(1)
    if self.normalize:
      if self.training:
        # In training we output log-softmax for NLL
        scores = F.log_softmax(s)
      else:
        # ...Otherwise 0-1 probabilities
        scores = F.softmax(s)
    else:
      scores = a.exp()
    return res, scores

  def forward(self, x, y, x_mask, y_mask):
    hiddens = self.init_hiddens(y, y_mask)
    c, start_scores = self.pointer(x, hiddens, x_mask)
    c_ = F.dropout(c, p=self.dropout_rate, training=self.training)
    hiddens = self.cell(c_, hiddens)
    c, end_scores = self.pointer(x, hiddens, x_mask)
    return start_scores, end_scores


class MemoryAnsPointer(nn.Module):

  def __init__(self,
               x_size,
               y_size,
               hidden_size,
               hop=1,
               dropout_rate=0,
               normalize=True):
    super(MemoryAnsPointer, self).__init__()
    self.normalize = normalize
    self.hidden_size = hidden_size
    self.hop = hop
    self.dropout_rate = dropout_rate
    self.FFNs_start = nn.ModuleList()
    self.SFUs_start = nn.ModuleList()
    self.FFNs_end = nn.ModuleList()
    self.SFUs_end = nn.ModuleList()
    for i in range(self.hop):
      self.FFNs_start.append(
          FeedForwardNetwork(x_size + y_size + 2 * hidden_size, hidden_size, 1,
                             dropout_rate))
      self.SFUs_start.append(SFU(y_size, 2 * hidden_size))
      self.FFNs_end.append(
          FeedForwardNetwork(x_size + y_size + 2 * hidden_size, hidden_size, 1,
                             dropout_rate))
      self.SFUs_end.append(SFU(y_size, 2 * hidden_size))

  def forward(self, x, y, x_mask, y_mask):
    z_s = y[:, -1, :].unsqueeze(1)  # [B, 1, I]
    z_e = None
    s = None
    e = None
    p_s = None
    p_e = None

    for i in range(self.hop):
      z_s_ = z_s.repeat(1, x.size(1), 1)  # [B, S, I]
      s = self.FFNs_start[i](torch.cat([x, z_s_, x * z_s_], 2)).squeeze(2)
      s.data.masked_fill_(x_mask.data, -float('inf'))
      p_s = F.softmax(s, dim=1)  # [B, S]
      u_s = p_s.unsqueeze(1).bmm(x)  # [B, 1, I]
      z_e = self.SFUs_start[i](z_s, u_s)  # [B, 1, I]
      z_e_ = z_e.repeat(1, x.size(1), 1)  # [B, S, I]
      e = self.FFNs_end[i](torch.cat([x, z_e_, x * z_e_], 2)).squeeze(2)
      e.data.masked_fill_(x_mask.data, -float('inf'))
      p_e = F.softmax(e, dim=1)  # [B, S]
      u_e = p_e.unsqueeze(1).bmm(x)  # [B, 1, I]
      z_s = self.SFUs_end[i](z_e, u_e)
    if self.normalize:
      if self.training:
        # In training we output log-softmax for NLL
        p_s = F.log_softmax(s, dim=1)  # [B, S]
        p_e = F.log_softmax(e, dim=1)  # [B, S]
      else:
        # ...Otherwise 0-1 probabilities
        p_s = F.softmax(s, dim=1)  # [B, S]
        p_e = F.softmax(e, dim=1)  # [B, S]
    else:
      p_s = s.exp()
      p_e = e.exp()
    return p_s, p_e


# ------------------------------------------------------------------------------
# Attentions
# ------------------------------------------------------------------------------


class SeqAttnMatch(nn.Module):
  """Given sequences X and Y, match sequence Y to each element in X.

    * o_i = sum(alpha_j * y_j) for i in X
    * alpha_j = softmax(y_j * x_i)
    """

  def __init__(self, input_size, identity=False):
    super(SeqAttnMatch, self).__init__()
    if not identity:
      self.linear = nn.Linear(input_size, input_size)
    else:
      self.linear = None

  def forward(self, x, y, y_mask=None):
    """
        Args:
            x: batch * len1 * hdim
            y: batch * len2 * hdim
            y_mask: batch * len2 (1 for padding, 0 for true)
        Output:
            matched_seq: batch * len1 * hdim
        """
    # Project vectors
    if self.linear:
      x_proj = self.linear(x.view(-1, x.size(2))).view(x.size())
      x_proj = F.relu(x_proj)
      y_proj = self.linear(y.view(-1, y.size(2))).view(y.size())
      y_proj = F.relu(y_proj)
    else:
      x_proj = x
      y_proj = y

    # Compute scores
    scores = x_proj.bmm(y_proj.transpose(2, 1))

    # Mask padding
    y_mask = y_mask.unsqueeze(1).expand(scores.size())
    scores.data.masked_fill_(y_mask.data, -float('inf'))

    # Normalize with softmax
    alpha = F.softmax(scores, dim=2)

    # Take weighted average
    matched_seq = alpha.bmm(y)
    return matched_seq


# like tf version DotAttention is rnet
class DotAttention(nn.Module):
  """Given sequences X and Y, match sequence Y to each element in X.

    * o_i = sum(alpha_j * y_j) for i in X
    * alpha_j = softmax(y_j * x_i)
    """

  def __init__(self,
               input_size,
               input_size2,
               hidden,
               dropout_rate=0.,
               combiner='gate'):
    super(DotAttention, self).__init__()
    self.linear = nn.Linear(input_size, hidden)
    self.linear2 = nn.Linear(input_size2, hidden)

    self.hidden = hidden
    self.combiner = combiner
    if combiner is not None:
      if combiner == 'gate':
        self.combine = Gate(input_size + input_size2, dropout_rate=dropout_rate)
        self.output_size = input_size + input_size2
      elif combiner == 'sfu':
        if input_size != input_size2:
          self.sfu_linear = nn.Linear(input_size2, input_size)
        else:
          self.sfu_linear = None
        self.combine = SFU(input_size,
                           3 * input_size,
                           dropout_rate=dropout_rate)
        self.output_size = input_size
      else:
        raise ValueError(f'not support {combiner}')

  def forward(self, x, y, y_mask=None):
    """
        Args:
            x: batch * len1 * hdim
            y: batch * len2 * hdim
            y_mask: batch * len2 (1 for padding, 0 for true)
        Output:
            matched_seq: batch * len1 * hdim
        """
    # Project vectors
    x_proj = self.linear(x.view(-1, x.size(2))).view(x.size(0), x.size(1),
                                                     self.hidden)
    x_proj = F.relu(x_proj)
    y_proj = self.linear2(y.view(-1, y.size(2))).view(y.size(0), y.size(1),
                                                      self.hidden)
    y_proj = F.relu(y_proj)

    #print(x_proj.shape, y_proj.shape)
    # Compute scores
    scores = x_proj.bmm(y_proj.transpose(2, 1))

    #print(scores.shape)

    #print(y_mask.shape)

    # Mask padding
    y_mask = y_mask.unsqueeze(1).expand(scores.size())
    scores.data.masked_fill_(y_mask.data, -float('inf'))

    # Normalize with softmax
    alpha = F.softmax(scores, dim=2)

    # Take weighted average
    matched_seq = alpha.bmm(y)

    if self.combiner is None:
      return matched_seq
    else:
      # TODO FIXME... why is False == True  ???
      #print(self.combiner, self.combiner is 'sfu', self.combiner == 'sfu')
      if self.combiner == 'gate':
        return self.combine(torch.cat([x, matched_seq], 2))
      elif self.combiner == 'sfu':
        if self.sfu_linear is not None:
          matched_seq = self.sfu_linear(matched_seq)

        return self.combine(
            x, torch.cat([matched_seq, x * matched_seq, x - matched_seq], 2))


class SelfAttnMatch(nn.Module):
  """Given sequences X and Y, match sequence Y to each element in X.

    * o_i = sum(alpha_j * x_j) for i in X
    * alpha_j = softmax(x_j * x_i)
    """

  def __init__(self, input_size, identity=False, diag=True):
    super(SelfAttnMatch, self).__init__()
    if not identity:
      self.linear = nn.Linear(input_size, input_size)
    else:
      self.linear = None
    self.diag = diag

  def forward(self, x, x_mask=None):
    """
        Args:
            x: batch * len1 * dim1
            x_mask: batch * len1 (1 for padding, 0 for true)
        Output:
            matched_seq: batch * len1 * dim1
        """
    # Project vectors
    if self.linear:
      x_proj = self.linear(x.view(-1, x.size(2))).view(x.size())
      x_proj = F.relu(x_proj)
    else:
      x_proj = x

    # Compute scores
    scores = x_proj.bmm(x_proj.transpose(2, 1))
    if not self.diag:
      x_len = x.size(1)
      for i in range(x_len):
        scores[:, i, i] = 0

    # Mask padding
    x_mask = x_mask.unsqueeze(1).expand(scores.size())
    scores.data.masked_fill_(x_mask.data, -float('inf'))

    # Normalize with softmax
    alpha = F.softmax(scores, dim=2)

    #print(scores.shape, ','.join([str(x) for x in alpha.view(-1).detach().cpu().numpy()]), sep='\t')

    # Take weighted average
    matched_seq = alpha.bmm(x)
    return matched_seq


class BilinearSeqAttn(nn.Module):
  """A bilinear attention layer over a sequence X w.r.t y:

    * o_i = softmax(x_i'Wy) for x_i in X.

    Optionally don't normalize output weights.
    """

  def __init__(self, x_size, y_size, identity=False, normalize=True):
    super(BilinearSeqAttn, self).__init__()
    self.normalize = normalize

    # If identity is true, we just use a dot product without transformation.
    if not identity:
      self.linear = nn.Linear(y_size, x_size)
    else:
      self.linear = None

  def forward(self, x, y, x_mask=None):
    """
        Args:
            x: batch * len * hdim1
            y: batch * hdim2
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            alpha = batch * len
        """
    Wy = self.linear(y) if self.linear is not None else y
    xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2)
    xWy.data.masked_fill_(x_mask.data, -float('inf'))
    if self.normalize:
      if self.training:
        # In training we output log-softmax for NLL
        alpha = F.log_softmax(xWy)
      else:
        # ...Otherwise 0-1 probabilities
        alpha = F.softmax(xWy)
    else:
      alpha = xWy.exp()
    return alpha


class LinearSeqAttn(nn.Module):
  """Self attention over a sequence:

    * o_i = softmax(Wx_i) for x_i in X.
    """

  def __init__(self, input_size):
    super(LinearSeqAttn, self).__init__()
    self.linear = nn.Linear(input_size, 1)

  def forward(self, x, x_mask=None):
    """
        Args:
            x: batch * len * hdim
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            alpha: batch * len
        """
    x_flat = x.view(-1, x.size(-1))
    scores = self.linear(x_flat).view(x.size(0), x.size(1))
    scores.data.masked_fill_(x_mask.data, -float('inf'))
    alpha = F.softmax(scores)
    return alpha


class NonLinearSeqAttn(nn.Module):
  """Self attention over a sequence:

    * o_i = softmax(function(Wx_i)) for x_i in X.
    """

  def __init__(self, input_size, hidden_size=128, activation='relu'):
    super(NonLinearSeqAttn, self).__init__()
    self.FFN = FeedForwardNetwork(input_size, hidden_size, 1, activation=activation)

  def forward(self, x, x_mask=None):
    """
        Args:
            x: batch * len * dim
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            alpha: batch * len
        """
    scores = self.FFN(x).squeeze(2)
    scores.data.masked_fill_(x_mask.data, -float('inf'))
    alpha = F.softmax(scores)
    return alpha


class MaxPooling(nn.Module):

  def forward(self, x, x_mask=None):
    if x_mask is None or x_mask.data.sum() == 0:
      return torch.max(x, 1)[0]
    else:
      x_mask = x_mask.unsqueeze(-1).expand(x.size())
      # will cause nan why TODO FIXME 
      # x.data.masked_fill_(x_mask.data, -float('inf'))
      x.data.masked_fill_(x_mask.data, -100)
      return torch.max(x, 1)[0]
      # # https://medium.com/@sonicboom8/sentiment-analysis-with-variable-length-sequences-in-pytorch-6241635ae130
      # # shold be same as above or use F.adaptive_max_pool1d and slightly slower
      # lengths = (1 - x_mask).sum(1)
      # return torch.cat([torch.max(i[:l], dim=0)[0].view(1,-1) for i,l in zip(x, lengths)], dim=0)


class SumPooling(nn.Module):

  def forward(self, x, x_mask=None):
    if x_mask is None or x_mask.data.sum() == 0:
      return torch.sum(x, 1)
    else:
      # lengths = (1 - x_mask).sum(1)
      # return torch.cat([torch.sum(i[:l], dim=0).view(1,-1) for i,l in zip(x, lengths)], dim=0)
      x_mask = x_mask.unsqueeze(-1).expand(x.size())
      x.data.masked_fill_(x_mask.data, 0.)
      return torch.sum(x, 1)


class MeanPooling(nn.Module):

  def forward(self, x, x_mask=None):
    if x_mask is None or x_mask.data.sum() == 0:
      return torch.mean(x, 1)
    else:
      # lengths = (1 - x_mask).sum(1)
      # return torch.cat([torch.sum(i[:l], dim=0).view(1,-1) for i,l in zip(x, lengths)], dim=0)
      x_mask = x_mask.unsqueeze(-1).expand(x.size())
      x.data.masked_fill_(x_mask.data, 0.)
      return torch.sum(x, 1) / torch.sum(1. - x_mask.float(), 1)


class MeanPooling2(nn.Module):

  def forward(self, x, x_mask=None):
    if x_mask is None or x_mask.data.sum() == 0:
      return torch.mean(x, 1)
    else:
      # lengths = (1 - x_mask).sum(1)
      # return torch.cat([torch.sum(i[:l], dim=0).view(1,-1) for i,l in zip(x, lengths)], dim=0)
      x_mask = x_mask.unsqueeze(-1).expand(x.size())
      x.data.masked_fill_(x_mask.data, 0.)
      return torch.mean(x, 1)


class ConcatPooling(nn.Module):

  def forward(self, x, x_mask=None):
    return x.view(-1, x.shape[1] * x.shape[2])


class DotPooling(nn.Module):
  """
    https://github.com/facebookresearch/dlrm/blob/master/dlrm_s_pytorch.py
    itersect_features
    """

  def __init__(self, interaction_itself=False, remove_duplicate=True):
    super(DotPooling, self).__init__()
    self.interaction_itself = interaction_itself
    self.remove_duplicate = remove_duplicate

  def forward(self, x, x_mask=None):
    batch_size = x.shape[0]
    T = x
    # perform a dot product
    Z = torch.bmm(T, torch.transpose(T, 1, 2))
    if not self.remove_duplicate:
      return Z.view((batch_size, -1))
    else:
      # Unlike tf remove duplicate does not make much speed difference
      # append dense feature with the interactions (into a row vector)
      # approach 1: all
      # Zflat = Z.view((batch_size, -1))
      # approach 2: unique
      _, ni, nj = Z.shape
      # approach 1: tril_indices
      # offset = 0 if self.arch_interaction_itself else -1
      # li, lj = torch.tril_indices(ni, nj, offset=offset)
      # approach 2: custom
      offset = 1 if self.interaction_itself else 0
      li = torch.tensor([i for i in range(ni) for j in range(i + offset)])
      lj = torch.tensor([j for i in range(nj) for j in range(i + offset)])
      Zflat = Z[:, li, lj]
      return Zflat


class TopKPooling(nn.Module):

  def __init__(self, top_k=2):
    super(TopKPooling, self).__init__()
    self.top_k = top_k

  def forward(self, x, x_mask=None):
    if x_mask is None or x_mask.data.sum() == 0:
      return torch.topk(x, k=self.top_k, dim=1)[0].view(x.size(0), -1)
    else:
      # x_mask = x_mask.unsqueeze(-1).expand(x.size())
      # x.data.masked_fill_(x_mask.data, -float('inf'))
      # return torch.topk(x, k=self.top_k, dim=1)[0].view(x.size(0), -1)
      #FIXME below not ok
      #return F.adaptive_max_pool1d(x, self.top_k).view(x.size(0), -1)
      # https://medium.com/@sonicboom8/sentiment-analysis-with-variable-length-sequences-in-pytorch-6241635ae130
      lengths = (1 - x_mask).sum(1)
      return torch.cat([
          torch.topk(i[:l], k=self.top_k, dim=0)[0].view(1, -1)
          for i, l in zip(x, lengths)
      ],
                       dim=0).view(x.size(0), -1)


class FMPooling(nn.Module):

  def __init__(self):
    super(FMPooling, self).__init__()
    self.sum_pooling = SumPooling()

  def forward(self, x, x_mask=None):
    summed_emb = self.sum_pooling(x, x_mask)
    summed_emb_square = summed_emb**2

    squared_emb = x**2
    squared_sum_emb = self.sum_pooling(squared_emb, x_mask)

    # [None * K]
    y_second_order = 0.5 * (summed_emb_square - squared_sum_emb)
    return y_second_order


class LastPooling(nn.Module):

  def __init__(self):
    super(LastPooling, self).__init__()

  def forward(self, x, x_mask=None):
    if x_mask is None or x_mask.data.sum() == 0:
      return x[:, -1, :]
    else:
      lengths = (1 - x_mask).sum(1)
      return torch.cat([i[l - 1, :] for i, l in zip(x, lengths)],
                       dim=0).view(x.size(0), -1)


class LinearSeqAttnPooling(nn.Module):
  """Self attention over a sequence:

    * o_i = softmax(Wx_i) for x_i in X.
    """

  def __init__(self, input_size):
    super(LinearSeqAttnPooling, self).__init__()
    self.linear = nn.Linear(input_size, 1)

  def forward(self, x, x_mask=None):
    """
        Args:
            x: batch * len * hdim
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            alpha: batch * len
        """
    # TODO why need contiguous
    x = x.contiguous()
    x_flat = x.view(-1, x.size(-1))
    scores = self.linear(x_flat).view(x.size(0), x.size(1))
    if x_mask is not None:
      scores.data.masked_fill_(x_mask.data, -float('inf'))
    alpha = F.softmax(scores, dim=-1)
    self.alpha = alpha
    return alpha.unsqueeze(1).bmm(x).squeeze(1)


class LinearSeqAttnPoolings(nn.Module):
  """Self attention over a sequence:

    * o_i = softmax(Wx_i) for x_i in X.
    """

  def __init__(self, input_size, num_poolings):
    super(LinearSeqAttnPoolings, self).__init__()
    self.num_poolings = num_poolings
    self.linear = nn.Linear(input_size, num_poolings)

  def forward(self, x, x_mask=None):
    """
        Args:
            x: batch * len * hdim
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            alpha: batch * len
        """
    # TODO why need contiguous
    x_mask = lele.tile(x_mask.unsqueeze(-1), -1, self.num_poolings)
    x = x.contiguous()
    x_flat = x.view(-1, x.size(-1))
    scores = self.linear(x_flat).view(x.size(0), x.size(1), self.num_poolings)
    scores = scores.transpose(-2, -1)
    if x_mask is not None:
      scores.data.masked_fill_(x_mask.data, -float('inf'))
    alpha = F.softmax(scores, dim=-1)
    self.alpha = alpha
    return alpha.bmm(x)


class NonLinearSeqAttnPooling(nn.Module):
  """Self attention over a sequence:

    * o_i = softmax(function(Wx_i)) for x_i in X.
    """

  def __init__(self, input_size, hidden_size=128, activation='relu'):
    super(NonLinearSeqAttnPooling, self).__init__()
    self.FFN = FeedForwardNetwork(input_size, hidden_size, 1, activation=activation)

  def forward(self, x, x_mask=None):
    """
        Args:
            x: batch * len * dim
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            alpha: batch * len
        """
    scores = self.FFN(x).squeeze(2)
    if x_mask is not None:
      scores.data.masked_fill_(x_mask.data, -float('inf'))
    alpha = F.softmax(scores, dim=-1)
    self.alpha = alpha
    return alpha.unsqueeze(1).bmm(x).squeeze(1)


class NonLinearSeqAttnPoolings(nn.Module):
  """Self attention over a sequence:

    * o_i = softmax(function(Wx_i)) for x_i in X.
    """

  def __init__(self, input_size, num_poolings, hidden_size=128):
    super(NonLinearSeqAttnPoolings, self).__init__()
    self.num_poolings = num_poolings
    self.FFN = FeedForwardNetwork(input_size, hidden_size, num_poolings)

  def forward(self, x, x_mask=None):
    """
        Args:
            x: batch * len * dim
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            alpha: batch * len
        """
    scores = self.FFN(x)
    x_mask = lele.tile(x_mask.unsqueeze(-1), -1, self.num_poolings)
    scores.data.masked_fill_(x_mask.data, -float('inf'))
    scores = scores.transpose(-2, -1)
    alpha = F.softmax(scores, dim=-1)
    self.alpha = alpha
    return alpha.bmm(x)


class ClsPooling(nn.Module):

  def __init__(self):
    super(ClsPooling, self).__init__()

  def forward(self, x, x_mask=None):
    return x[:, 0]


class Pooling(nn.Module):

  def __init__(self,
               name,
               input_size=0,
               output_size=None,
               top_k=2,
               att_activation=F.relu,
               **kwargs):
    super(Pooling, self).__init__(**kwargs)

    self.top_k = top_k

    self.input_size = input_size

    self.poolings = nn.ModuleList()

    def _get_pooling(name):
      if name == 'max':
        return MaxPooling()
      elif name == 'mean':
        return MeanPooling()
      elif name == 'mean2':
        return MeanPooling2()
      elif name == 'sum':
        return SumPooling()
      elif name == 'attention' or name == 'att':
        return NonLinearSeqAttnPooling(input_size)
      elif name == 'tanh_att':
        return NonLinearSeqAttnPooling(input_size, activation='tanh')
      elif name == 'attention2' or name == 'att2':
        return NonLinearSeqAttnPooling(input_size, int(input_size / 2))
      elif name == 'tanh_att2':
        return NonLinearSeqAttnPooling(input_size, int(input_size / 2), activation='tanh')
      elif name == 'linear_attention' or name == 'linear_att' or name == 'latt':
        return LinearSeqAttnPooling(input_size)
      elif name == 'topk' or name == 'top_k':
        return TopKPooling(top_k=top_k)
      elif name == 'first' or name == 'cls':
        return ClsPooling()
      elif name == 'last':
        return LastPooling()
      elif name == 'concat' or name == 'cat':
        return ConcatPooling()
      elif name == 'fm':
        return FMPooling()
      elif name == 'dot':
        return DotPooling()
      else:
        raise f'Unsupport pooling now:{name}'

    self.names = name.split(',')
    for name in self.names:
      self.poolings.append(_get_pooling(name))

    self.output_sizes = {}
    self._set_output_size()

    #logging.info('poolings:', self.poolings)

  def _set_output_size(self):
    self.output_size = 0
    for name in self.names:
      if name in self.output_sizes:
        self.output_size = self.output_sizes[name]
      else:
        if name == 'topk' or name == 'top_k':
          self.output_size += self.input_size * self.top_k
        else:
          self.output_size += self.input_size
    self.output_dim = self.output_size

  def set_output_size(self, output_size=None, name=None):
    if not output_size:
      self._set_output_size()
    if not name:
      self.output_size = output_size
      self.output_dim = output_size
    else:
      self.output_sizes[name] = output_size
      self._set_output_size()

  def forward(self, x, mask=None, calc_word_scores=False):
    results = []
    self.word_scores = []
    if mask is not None:
      # *mask.max(-1, keepdim=True)[0] to avoid all position mask -> nan
      mask = ((1 - mask) * mask.max(-1, keepdim=True)[0]).bool()
    for i, pooling in enumerate(self.poolings):
      results.append(pooling(x, mask))
      # ic(results[-1], results[-1].min(), results[-1].max())
      if calc_word_scores:
        import melt
        # TODO remove from melt
        self.word_scores.append(
            melt.get_words_importance(outputs,
                                      sequence_length,
                                      top_k=self.top_k,
                                      method=self.names[i]))

    return torch.cat(results, -1)


Poolings = Pooling


# TODO can we do multiple exclusive linear simultaneously ?
class Linears(nn.Module):

  def __init__(self, input_size, output_size, num, **kwargs):
    super(Linears, self).__init__(**kwargs)
    self.num = num
    self.linears = lele.clones(nn.Linear(input_size, output_size), num)

  def forward(self, x):
    inputs = x.split(1, 1)
    results = []
    for linear, x in zip(self.linears, inputs):
      result = linear(x)
      results.append(result)
    result = torch.cat(results, 1)
    return result


# ------------------------------------------------------------------------------
# Functional Units
# ------------------------------------------------------------------------------


class Gate(nn.Module):
  """Gate Unit
    g = sigmoid(Wx)
    x = g * x
    """

  def __init__(self, input_size, dropout_rate=0.):
    super(Gate, self).__init__()
    self.linear = nn.Linear(input_size, input_size, bias=False)
    self.dropout_rate = dropout_rate

  def forward(self, x):
    """
        Args:
            x: batch * len * dim
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            res: batch * len * dim
        """
    if self.dropout_rate:
      x = F.dropout(x, p=self.dropout_rate, training=self.training)
    x_proj = self.linear(x)
    gate = torch.sigmoid(x)
    return x_proj * gate


class SFU(nn.Module):
  """Semantic Fusion Unit
    The ouput vector is expected to not only retrieve correlative information from fusion vectors,
    but also retain partly unchange as the input vector
    """

  def __init__(self, input_size, fusion_size, dropout_rate=0.):
    super(SFU, self).__init__()
    self.linear_r = nn.Linear(input_size + fusion_size, input_size)
    self.linear_g = nn.Linear(input_size + fusion_size, input_size)
    self.dropout_rate = dropout_rate

  def forward(self, x, fusions):
    r_f = torch.cat([x, fusions], 2)
    if self.dropout_rate:
      r_f = F.dropout(r_f, p=self.dropout_rate, training=self.training)
    r = torch.tanh(self.linear_r(r_f))
    g = torch.sigmoid(self.linear_g(r_f))
    o = g * r + (1 - g) * x
    return o


class SFUCombiner(nn.Module):
  """Semantic Fusion Unit
    The ouput vector is expected to not only retrieve correlative information from fusion vectors,
    but also retain partly unchange as the input vector
    """

  def __init__(self, input_size, fusion_size, dropout_rate=0.):
    super(SFUCombiner, self).__init__()
    self.linear_r = nn.Linear(input_size + fusion_size, input_size)
    self.linear_g = nn.Linear(input_size + fusion_size, input_size)
    self.dropout_rate = dropout_rate

  def forward(self, x, y):
    r_f = torch.cat([x, y, x * y, x - y], 2)
    if self.dropout_rate:
      r_f = F.dropout(r_f, p=self.dropout_rate, training=self.training)
    r = torch.tanh(self.linear_r(r_f))
    g = torch.sigmoid(self.linear_g(r_f))
    o = g * r + (1 - g) * x
    return o


def dense_layer(inp, out, dropout=0., activation='ReLU'):
  assert inp > 0 and out > 0, f'{inp} {out}'
  activate = getattr(torch.nn.modules.activation, activation)
  return nn.Sequential(nn.Linear(inp, out), activate(),
                       nn.Dropout(dropout))


class MLP(nn.Module):

  def __init__(self,
               input_dim,
               dims,
               output_dim=None,
               dropout=0.,
               activation='ReLU'):
    super(MLP, self).__init__()
    dims = dims.copy()
    num_layers = len(dims)
    assert num_layers >= 1
    if not isinstance(dropout, (list, tuple)):
      dropout = [dropout] * num_layers
    if not isinstance(activation, (list, tuple)):
      activation = [activation] * num_layers

    dims.insert(0, input_dim)

    def _get_dim(dims, i):
      if dims[i]:
        return dims[i]
      return dims[i - 1] // 2

    for i in range(1, len(dims)):
      dims[i] = _get_dim(dims, i)
    self.dense = nn.Sequential()
    for i in range(num_layers):
      self.dense.add_module(
          'dense_layer_{}'.format(i),
          dense_layer(dims[i], dims[i + 1], dropout[i], activation[i]))
    if output_dim is not None:
      self.dense.add_module('last_linear', nn.Linear(dims[-1], output_dim))
      self.output_dim = output_dim
    else:
      self.output_dim = dims[-1]
    self.num_layers = num_layers

  def forward(self, x):
    return self.dense(x)


# TODO without using Embedding ?
class LookupArray(nn.Module):

  def __init__(self, nparray):
    super(LookupArray, self).__init__()
    self.input_dim = nparray.shape[0]
    self.embedding = nn.Embedding(self.input_dim, 1)
    self.embedding.weight.data.copy_(torch.as_tensor(nparray).view(-1, 1))
    self.embedding.weight.requires_grad = False
    # self.embedding = nn.Embedding.from_pretrained(torch.as_tensor(nparray).view(-1, 1))

  def forward(self, x):
    x = x % self.input_dim
    return self.embedding(x).squeeze(-1).long()


class MultiSample(nn.Module):

  def __init__(self,
               input_dim,
               output_dim=None,
               dims=[],
               drop_rate=0.1,
               num_experts=5,
               activation='ReLU',
               **kwargs):
    super().__init__(**kwargs)
    self.num_experts = num_experts
    if output_dim is not None:
      dims.append(output_dim)
    self.mlps = nn.ModuleList([
        MLP(input_dim, dims, output_dim=output_dim, activation=activation)
        for _ in range(num_experts)
    ])
    # weights = [x + 1 for x in range(num_experts)]
    weights = [1. for x in range(num_experts)]
    self.dropouts = nn.ModuleList(
        [nn.Dropout(drop_rate * weights[i]) for i in range(num_experts)])

  def forward(self, x, reduce=True):
    xs = []
    for i in range(self.num_experts):
      x_i = self.dropouts[i](x)
      x_i = self.mlps[i](x_i)
      xs += [x_i]
    ret = torch.stack(xs, axis=1).mean(axis=1)
    gezi.set('xs', xs)
    return ret


class MultiDropout(nn.Module):

  def __init__(self,
               input_dim,
               output_dim,
               dims=[],
               drop_rate=0.1,
               num_experts=5,
              #  activation='ReLU',
               **kwargs):
    super().__init__(**kwargs)
    self.num_experts = num_experts
    if output_dim is not None:
      dims.append(output_dim)
    # self.mlp = MLP(input_dim, dims, output_dim=output_dim, activation=activation)
    self.mlp = nn.Linear(input_dim, output_dim)
    weights = [x + 1 for x in range(num_experts)]
    # weights = [1. for x in range(num_experts)]
    self.dropouts = nn.ModuleList(
        [nn.Dropout(drop_rate *  weights[i]) for i in range(num_experts)])

  def forward(self, x, reduce=True):
    xs = []
    for i in range(self.num_experts):
      x_i = self.dropouts[i](x)
      x_i = self.mlp(x_i)
      xs += [x_i]
    ret = torch.stack(xs, axis=1).mean(axis=1)
    gezi.set('xs', xs)
    return ret


class MMoE(nn.Module):

  def __init__(self,
               num_tasks,
               input_dim,
               dims=[],
               output_dim=None,
               num_experts=None,
               activation='ReLU',
               return_att=False):
    super().__init__()
    self.num_tasks = num_tasks
    self.num_experts = num_experts if num_experts else num_tasks * 2 + 1
    self.mlps = nn.ModuleList([
        MLP(input_dim, dims, output_dim=output_dim, activation=activation)
        for _ in range(num_experts)
    ])
    if output_dim is None:
      output_dim = dims[-1]
    idim = output_dim
    self.gates = nn.ModuleList([
        nn.Sequential(nn.Linear(idim, num_experts, bias=False), nn.Softmax())
        for _ in range(num_tasks)
    ])
    self.return_att = return_att

  def call(self, x):
    # [bs, hidden] * n
    outputs = [mlp(x) for mlp in self.mlps]
    # [bs, n, hidden]
    outputs = torch.stack(outputs, 1)
    res = []
    atts = []
    for i in range(self.num_tasks):
      probs = self.gates[i](x)
      probs = probs.unsqueeze(-1)
      outputs_ = outputs * probs
      output = outputs_.sum(1)
      res.append(output)
      atts.append(probs)

    if self.return_att:
      return res, atts
    else:
      return res


# ------------------------------------------------------------------------------
# Functional TODO move to ops.py
# ------------------------------------------------------------------------------


def Embedding(num_embeddings, embedding_dim, **kwargs):
  emb = nn.Embedding(num_embeddings, embedding_dim, **kwargs)
  # similar as tf.kears.layers.Embedding init
  torch.nn.init.uniform_(emb.weight, a=-0.05, b=0.05)
  return emb


def uniform_weights(x, x_mask):
  """Return uniform weights over non-masked x (a sequence of vectors).

    Args:
        x: batch * len * hdim
        x_mask: batch * len (1 for padding, 0 for true)
    Output:
        x_avg: batch * hdim
    """
  alpha = Variable(torch.ones(x.size(0), x.size(1)))
  if x.data.is_cuda:
    alpha = alpha.cuda()
  alpha = alpha * x_mask.eq(0).float()
  alpha = alpha / alpha.sum(1).expand(alpha.size())
  return alpha


def weighted_avg(x, weights):
  """Return a weighted average of x (a sequence of vectors).

    Args:
        x: batch * len * hdim
        weights: batch * len, sum(dim = 1) = 1
    Output:
        x_avg: batch * hdim
    """
  return weights.unsqueeze(1).bmm(x).squeeze(1)


class Dice(nn.Module):

  def __init__(self, feature_num):
    super(Dice, self).__init__()
    self.alpha = nn.Parameter(torch.ones(1))
    self.feature_num = feature_num
    self.bn = nn.BatchNorm1d(self.feature_num, affine=False)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    x_norm = self.bn(x)
    x_p = self.sigmoid(x_norm)

    return self.alpha * (1.0 - x_p) * x + x_p * x


class SinusoidalPositionEmbedding(Module):
  """定义Sin-Cos位置Embedding
    """

  def __init__(self, output_dim, merge_mode='add', custom_position_ids=False):
    super(SinusoidalPositionEmbedding, self).__init__()
    self.output_dim = output_dim
    self.merge_mode = merge_mode
    self.custom_position_ids = custom_position_ids

  def forward(self, inputs):
    input_shape = inputs.shape
    _, seq_len = input_shape[0], input_shape[1]
    position_ids = torch.arange(seq_len).type(torch.float)[None]
    indices = torch.arange(self.output_dim // 2).type(torch.float)
    indices = torch.pow(10000.0, -2 * indices / self.output_dim)
    embeddings = torch.einsum('bn,d->bnd', position_ids, indices)
    embeddings = torch.stack(
        [torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
    embeddings = torch.reshape(embeddings, (-1, seq_len, self.output_dim))

    if self.merge_mode == 'add':
      return inputs + embeddings.to(inputs.device)
    elif self.merge_mode == 'mul':
      return inputs * (embeddings + 1.0).to(inputs.device)
    elif self.merge_mode == 'zero':
      return embeddings.to(inputs.device)


def sequence_masking(x, mask, value='-inf', axis=None):
  if mask is None:
    return x
  else:
    if value == '-inf':
      value = -1e12
    elif value == 'inf':
      value = 1e12
    assert axis > 0, 'axis must be greater than 0'
    for _ in range(axis - 1):
      mask = torch.unsqueeze(mask, 1)
    for _ in range(x.ndim - mask.ndim):
      mask = torch.unsqueeze(mask, mask.ndim)
    return x * mask + value * (1 - mask)


def add_mask_tril(logits, mask):
  if mask.dtype != logits.dtype:
    mask = mask.type(logits.dtype)
  logits = sequence_masking(logits, mask, '-inf', logits.ndim - 2)
  logits = sequence_masking(logits, mask, '-inf', logits.ndim - 1)
  # 排除下三角
  mask = torch.tril(torch.ones_like(logits), diagonal=-1)
  logits = logits - mask * 1e12
  return logits


class GlobalPointer(Module):
  """全局指针模块
    将序列的每个(start, end)作为整体来进行判断
    """

  def __init__(self, heads, head_size, hidden_size, RoPE=True):
    super(GlobalPointer, self).__init__()
    self.heads = heads
    self.head_size = head_size
    self.RoPE = RoPE
    self.dense = nn.Linear(hidden_size, self.head_size * self.heads * 2)


#     def reset_params(self):
#         nn.init.xavier_uniform_(self.dense.weight)

  def forward(self, inputs, mask=None):
    inputs = self.dense(inputs)
    inputs = torch.split(inputs, self.head_size * 2, dim=-1)
    # 按照-1这个维度去分，每块包含x个小块
    inputs = torch.stack(inputs, dim=-2)
    # 沿着一个新维度对输入张量序列进行连接。 序列中所有的张量都应该为相同形状
    qw, kw = inputs[..., :self.head_size], inputs[..., self.head_size:]
    # 分出qw和kw
    # RoPE编码
    if self.RoPE:
      pos = SinusoidalPositionEmbedding(self.head_size, 'zero')(inputs)
      cos_pos = pos[..., None, 1::2].repeat_interleave(2, dim=-1)
      sin_pos = pos[..., None, ::2].repeat_interleave(2, dim=-1)
      qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], 4)
      qw2 = torch.reshape(qw2, qw.shape)
      qw = qw * cos_pos + qw2 * sin_pos
      kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], 4)
      kw2 = torch.reshape(kw2, kw.shape)
      kw = kw * cos_pos + kw2 * sin_pos
    # 计算内积
    logits = torch.einsum('bmhd , bnhd -> bhmn', qw, kw)
    # 排除padding 排除下三角
    logits = add_mask_tril(logits, mask)

    # scale返回
    return logits / self.head_size**0.5


class EfficientGlobalPointer(Module):
  """全局指针模块
    将序列的每个(start, end)作为整体来进行判断
    """

  def __init__(self, heads, head_size, hidden_size, RoPE=True):
    super(EfficientGlobalPointer, self).__init__()
    self.heads = heads
    self.head_size = head_size
    self.RoPE = RoPE
    self.dense_1 = nn.Linear(hidden_size, self.head_size * 2)
    self.dense_2 = nn.Linear(self.head_size * 2, self.heads * 2)

  def forward(self, inputs, mask=None):
    inputs = self.dense_1(inputs)  # batch,
    # 沿着一个新维度对输入张量序列进行连接。 序列中所有的张量都应该为相同形状
    qw, kw = inputs[..., :self.head_size], inputs[..., self.head_size:]
    # 分出qw和kw
    # RoPE编码
    if self.RoPE:
      pos = SinusoidalPositionEmbedding(self.head_size, 'zero')(inputs)
      cos_pos = pos[..., 1::2].repeat_interleave(2, dim=-1)
      sin_pos = pos[..., ::2].repeat_interleave(2, dim=-1)
      qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], 3)
      qw2 = torch.reshape(qw2, qw.shape)
      qw = qw * cos_pos + qw2 * sin_pos
      kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], 3)
      kw2 = torch.reshape(kw2, kw.shape)
      kw = kw * cos_pos + kw2 * sin_pos
    # 计算内积
    logits = torch.einsum('bmd , bnd -> bmn', qw, kw) / self.head_size**0.5
    bias = torch.einsum('bnh -> bhn', self.dense_2(inputs)) / 2
    logits = logits[:, None] + bias[:, :self.heads,
                                    None] + bias[:, self.heads:, :, None]
    # 排除padding 排除下三角
    logits = add_mask_tril(logits, mask)

    # scale返回
    return logits


class TimeDistributed(nn.Module):

  def __init__(self, model, batch_first=True):
    super(TimeDistributed, self).__init__()
    self.model = model
    self.batch_first = batch_first

  # (bs, timesteps, local_time_steps, input_size)
  def forward(self, x):
    if len(x.size()) <= 2:
      return self.model(x)

    # Squash samples and timesteps into a single axis
    x_reshape = x.contiguous().view(
        -1, x.size(-2), x.size(-1))  # (samples * timesteps, input_size)

    y = self.model(x_reshape)

    # We have to reshape Y
    y = y.contiguous().view(x.size(0), -1,
                            y.size(-1))  # (samples, timesteps, output_size)

    # IF need timesteps first,   We have to reshape Y
    if not self.batch_first:
      y = y.transpose(
          0, 1).contiguous()  # transpose to (timesteps, samples, output_size)
    return y
