v1 = 0
mns1 = [

]

v2 = 0
mns2 = [    
]

# v3 = 4
# mns3 = [    
#    'deberta-v3.len1280.flag-encode_targets-cls.concat_last.rank_loss_rate-0.1',
#    'deberta-v3.len1280.flag-encode_targets-cls.concat_last.rank_loss_rate-0.1.trans-nl-ft3',
#    'deberta-v3.len1280.flag-encode_targets-cls.concat_last.rank_loss_rate-0.1.trans-fr-ft2',
#   # 'deberta-v3.len1280.flag-encode_targets-cls.concat_last.rank_loss_rate-0.1.trans-es-ft4',
#   #  'deberta-v3.len1280.flag-encode_targets-cls.concat_last.rank_loss_rate-0.1.trans-de-ft2',
#   #  'deberta-v3.len1280.flag-encode_targets-cls.concat_last.rank_loss_rate-0.1.trans-cn-ft4',
   
#   #  'deberta-v3.len1280.flag-encode_targets-cls',
# ]

v3 = 16
mns3 = [
  # 'deberta-v3-base.len1280.flag-encode_targets-cls-crank.lr-5e-5',
  # 'deberta-v3-base.len1280.flag-encode_targets-cls-crank.lr-5e-5.a',
  # 'deberta-v3-base.len1280.flag-encode_targets-cls.lr-5e-5',
  # 'deberta-v3.len1280.flag-encode_targets-cls-crank',
  # 'deberta-v3.len1280.flag-encode_targets-cls-crank.ft2',
  # 'deberta-v3.len1280.flag-encode_targets-cls',
  
  # 'deberta-v3.len1280.flag-encode_targets-cls.v1_only',
  # 'deberta-v3.len1280.flag-encode_targets-cls-crank.v1_ft.v1_dynamic',
  # 'deberta-v3-base.len1280.flag-encode_targets-cls.lr-5e-5.v1_only',
   
  # 'deberta-v3.len1280.flag-encode_targets-cls-crank.pre-trans-nl',
  # 'deberta-v3.len1280.flag-encode_targets-cls-crank.pre-trans-fr',
  # 'deberta-v3.len1280.flag-encode_targets-cls-crank.pre-trans-de',
  # 'deberta-v3.len1280.flag-encode_targets-cls-crank.pre-trans-pt',
  # 'deberta-v3.len1280.flag-encode_targets-cls-crank.pre-trans-af',
  # 'deberta-v3.len1280.flag-encode_targets-cls-crank.pre-trans-cn',
  # 'deberta-v3.len1280.flag-encode_targets-cls-crank.pre-trans-ru',
  # 'deberta-v3.len1280.flag-encode_targets-cls-crank.pre-trans-fi',
  # 'deberta-v3.len1280.flag-encode_targets-cls-crank.pre-trans-sv',
  # 'deberta-v3.len1280.flag-encode_targets-cls-crank.pre-trans-ja',
  # 'deberta-v3.len1280.flag-encode_targets-cls-crank.pre-trans-ko',
  # 'deberta-v3.len1280.flag-encode_targets-cls-crank.pre-trans-el',
  # 'deberta-v3.len1280.flag-encode_targets-cls-crank.pre-trans-hr',
  # 'deberta-v3.len1280.flag-encode_targets-cls-crank.pre-trans-cy',
  # 'deberta-v3.len1280.flag-encode_targets-cls.add_v1.ep-1',
    'deberta-v3.len1280.flag-encode_targets-cls-crank.add_v1.ep-1',
]


mns = mns1 + mns2 + mns3
v = v3

weights_dict = {}

# weights = [1] * len(mns)
def get_weight(x):
  #return weights_dict.get(x, 1)
  # return weights_dict[x]
  return 1

if all(x in weights_dict for x in mns):
  weights = [weights_dict[x] for x in mns]
else:
  weights = [get_weight(x) for x in mns]
weights = []
# weights = [1, 1, 0.5, 0.5, 0.5]
# weights = []
weights.extend([1] * 100)
ic(list(zip(mns, weights)), len(mns))

SAVE_PRED = 0
