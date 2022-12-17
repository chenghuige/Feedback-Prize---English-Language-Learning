#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   __init__.py
#        \author   chenghuige  
#          \date   2016-08-15 16:32:00.341661
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ["NCCL_DEBUG"] = 'WARNING'
import warnings
warnings.simplefilter("ignore") 

import time
if 'TZ' not in os.environ or os.environ['TZ'] != 'Asia/Shanghai':
  os.environ['TZ'] = 'Asia/Shanghai'
  time.tzset()

import gezi.utils 
from gezi.utils import *

try:
  import gezi.melt
  from gezi.melt import *
except Exception:
  print(traceback.format_exc(), file=sys.stderr)  

from gezi.timer import *
from gezi.nowarning import * 
from gezi.gezi_util import * 
from gezi.avg_score import *
from gezi.zhtools import *
from gezi.util import * 
from gezi.rank_metrics import *
from gezi.topn import *
from gezi.vocabulary import Vocabulary, Vocab
from gezi.word_counter import WordCounter
from gezi.ngram import *
from gezi.hash import *
from gezi.geneticalgorithm import geneticalgorithm

#if using baidu segmentor set encoding='gbk'
encoding='utf8' 
#encoding='gbk'

# try:
#   import matplotlib
#   matplotlib.use('Agg')
# except Exception:
#   pass 

import gezi.summary
from gezi.summary import SummaryWriter, EagerSummaryWriter

import traceback

from gezi.segment import *
#try:
#  from gezi.libgezi_util import *
#  import gezi.libgezi_util as libgezi_util
#  from gezi.segment import *
#  import gezi.bigdata_util
#except Exception:
#  print(traceback.format_exc(), file=sys.stderr)
#  print('import libgezi, segment bigdata_util fail')
#
#try:
#  from gezi.pydict import *
#except Exception:
#  #print(traceback.format_exc(), file=sys.stderr)
#  #print('import pydict fail')
#  pass

try:
  from gezi.libgezi_util import *
except Exception:
  print(traceback.format_exc(), file=sys.stderr)

try:
  import gezi.metrics
except Exception:
  print(traceback.format_exc(), file=sys.stderr) 

# TODO remove multiprocessing/context.py import ...
from gezi.util import Manager 

import gezi.plot
from gezi.plot import line, enable_plotly_in_cell


from shutil import make_archive
import gezi.common
