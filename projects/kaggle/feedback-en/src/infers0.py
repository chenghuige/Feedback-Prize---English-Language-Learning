#!/usr/bin/env python
# coding: utf-8

import sys, os
models = [
  'deberta-v3-base.len1280.flag-encode_targets-cls-crank.lr-5e-5',
  'deberta-v3-base.len1280.flag-encode_targets-cls-crank.lr-5e-5.a',
  'deberta-v3-base.len1280.flag-encode_targets-cls.lr-5e-5',
  'deberta-v3.len1280.flag-encode_targets-cls-crank',
  'deberta-v3.len1280.flag-encode_targets-cls-crank.ft2',
  'deberta-v3.len1280.flag-encode_targets-cls',
  'deberta-v3.len1280.flag-encode_targets-cls-crank.pre-trans-nl',
  'deberta-v3.len1280.flag-encode_targets-cls-crank.pre-trans-fr',
  'deberta-v3.len1280.flag-encode_targets-cls-crank.pre-trans-de',
  'deberta-v3.len1280.flag-encode_targets-cls-crank.pre-trans-pt',
  #'deberta-v3.len1280.flag-encode_targets-cls-crank.pre-trans-af',
  #'deberta-v3.len1280.flag-encode_targets-cls-crank.pre-trans-cn',
  #'deberta-v3.len1280.flag-encode_targets-cls-crank.pre-trans-ru',
  #'deberta-v3.len1280.flag-encode_targets-cls-crank.pre-trans-fi',
  #'deberta-v3.len1280.flag-encode_targets-cls-crank.pre-trans-sv',
  #'deberta-v3.len1280.flag-encode_targets-cls-crank.pre-trans-ja',
  #'deberta-v3.len1280.flag-encode_targets-cls-crank.pre-trans-ko',
  #'deberta-v3.len1280.flag-encode_targets-cls-crank.pre-trans-el',
  #'deberta-v3.len1280.flag-encode_targets-cls-crank.pre-trans-hr',
  #'deberta-v3.len1280.flag-encode_targets-cls-crank.pre-trans-cy',
  ]

for model in models:
  suffix = sys.argv[1]
  command = f'sh ./scripts/infer.sh {model} --{suffix}'
  print(command)
  os.system(command)
