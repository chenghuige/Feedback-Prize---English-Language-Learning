#!/usr/bin/env python
# coding: utf-8

# In[45]:


import sys
sys.path.append('..')
from gezi.common import *
from src.config import *
from src.preprocess import *
from src.eval import *
gezi.init_flags()


# In[117]:


lang = 'zh-cn' if gezi.in_notebook() else sys.argv[1]
lang_ = lang.split('-')[-1]
ic(lang, lang_)


# In[46]:


from nltk.tokenize import sent_tokenize
from googletrans import Translator
import time
trans = Translator()
ic(trans.translate('照片').text)


# In[99]:


def translate(text, lang='zh-cn', max_len=512):
  num_words = len(text.split())
  if num_words <= max_len:
    mid_text = trans.translate(text, dest=lang).text
    target_text = trans.translate(mid_text, dest='en').text
    time.sleep(0.5)
  else:
    mid_texts = []
    target_texts = []
    sents = sent_tokenize(text)    
    num_sents = len(sents)
    parts = int(num_words / max_len) + 1
    if num_sents > parts:
      sents = np.array_split(sents, parts)
      for i in range(len(sents)):
        sents[i] = ''.join(sents[i])
    for sent in sents:
      # print(sent)
      mid_text = trans.translate(sent, dest=lang).text
      # print(mid_text)
      target_text = trans.translate(mid_text, dest='en').text
      # print(target_text)
      time.sleep(0.5)
      mid_texts.append(mid_text)
      target_texts.append(target_text)
    mid_text = ''.join(mid_texts)
    target_text = ''.join(target_texts)
  return mid_text, target_text


# In[48]:


trans_dict = {}
trans_dict2 = {}


# In[102]:


def translate_(text, lang):
  if text not in trans_dict or text not in trans_dict2:
    try:
      mid_text, target_text = translate(text, lang)
      trans_dict[text] = mid_text
      trans_dict2[text] = target_text
    except Exception as e:
      # ic(e, len(text.split()))
      # bads.append(text)
      time.sleep(1)
      try:
        mid_text, target_text = translate(text, lang, 256)
        trans_dict[text] = mid_text
        trans_dict2[text] = target_text
      except Exception as e:
        time.sleep(1)
        try:
          mid_text, target_text = translate(text, lang, 128)
          trans_dict[text] = mid_text
          trans_dict2[text] = target_text
        except Exception as e:
          time.sleep(1)
          try:
            mid_text, target_text = translate(text, lang, 64)
            trans_dict[text] = mid_text
            trans_dict2[text] = target_text
          except Exception as e:
            ic(e, len(text.split()))


# In[116]:


d = pd.read_csv(f'{FLAGS.root}/train.csv')


# In[103]:


gezi.prun_loop(lambda x: translate_(x, lang), d.full_text.values, 1, desc='trans')
gezi.prun_loop(lambda x: translate_(x, lang), d.full_text.values, 1, desc='trans')


# In[ ]:

for text in d.full_text.values:
  if text not in trans_dict:
    trans_dict[text] = text
    trans_dict2[text] = text

ic(len(trans_dict), len(d))
assert len(trans_dict) == len(d)


# In[107]:


d[f'{lang_}_text'] = d.full_text.apply(lambda x: trans_dict[x])


# In[109]:


d[f'trans_{lang_}'] = d.full_text.apply(lambda x: trans_dict2[x])


# In[113]:


gezi.set_pd_widder()


# In[114]:


d


# In[115]:


d.to_csv(f'{FLAGS.root}/train_{lang_}.csv', index=False)


# In[ ]:




