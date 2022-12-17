#!/usr/bin/env python
# coding: utf-8

# # KAGGLE: Feedback Prize - English Language Learning
# 
# 

# In[ ]:


NAME_PROJECT = "Feedback Prize - English Language Learning"
import time

# Config
class CFG:
    model_name = "microsoft/deberta-v3-large"

    # hyperparameters
    transformer_lr     = 1e-6
    head_lr            = 3e-4
    batch_size         = 2
    
    accumulation_steps = 1
    warmup_epochs      = 0.33333333
    max_epochs         = 5
    adam_eps           = 1e-5
    scheduler          = "linear"
    max_grad_norm      = 1
    head_dropout       = 0.2
    rnn_dropout        = 0.

    # data split
    num_folds          = 5
    train_on_full_data = False
    current_fold       = 0

    # others
    # seed = int(time.time())
    seed                = 0
    eval_batch_size     = 8
    debug               = False
    fp16                = True
    gradient_checkpoint = False
    optimizer_bit8      = False
    use_augmentation    = False
    new_line_token      = True

    # AWP
    use_awp         = True
    adversarial_lr  = 1e-5
    adversarial_eps = 1e-3
    start_awp_epoch = 1
    awp_steps       = 1


IDS_TO_LABELS = {
    0 : "cohesion",
    1 : "syntax",
    2 : "vocabulary",
    3 : "phraseology",
    4 : "grammar",
    5 : "conventions"
}


# ### 1) Imports
# 

# In[ ]:


get_ipython().run_cell_magic('capture', '', 'import os\n\nIS_KAGGLE = bool(os.environ.get(\'KAGGLE_KERNEL_RUN_TYPE\', \'\'))\nIS_COLAB = bool(os.environ.get("COLAB_GPU"))\nIS_PAPER = bool(os.environ.get("PAPERSPACE_NOTEBOOK_ID"))\n\nif IS_COLAB:\n    !pip install wandb\n    !pip install transformers\n    !pip install sentencepiece\n    !pip install pytorch_lightning\n    !pip install iterative-stratification\n\nif IS_KAGGLE:\n    !pip install --upgrade transformers\n    !pip install iterative-stratification\n\nif IS_PAPER:\n    !pip install pytorch_lightning\n    !pip install wandb\n    !pip install kaggle\n    !pip install iterative-stratification\n\nos.environ["TOKENIZERS_PARALLELISM"] = "false"')


# In[ ]:


import random
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.checkpoint import checkpoint

import pytorch_lightning as pl
from sklearn.metrics import mean_squared_error

import transformers

import wandb

import warnings, logging
warnings.simplefilter('ignore')
logging.disable(logging.WARNING)


# In[ ]:


print(torch.__version__)


# In[ ]:


CFG.num_jobs = os.cpu_count()
CFG.device = "cuda" if torch.cuda.is_available() else "cpu"
CFG.device_name = "CPU"

if CFG.device == "cuda":
    CFG.device_name = torch.cuda.get_device_name(0)
    
    if CFG.optimizer_bit8:
        get_ipython().system('pip install bitsandbytes')
        import bitsandbytes as bnb

DBS_FROM_KAGGLE = [
    ("feedback-prize-english-language-learning", True),
    ("feedback-prize-effectiveness", True),
    ("feedback-prize-2021", True)
]


# In[ ]:


# SECRETS 
os.environ["KAGGLE_USERNAME"] = ""
os.environ["KAGGLE_KEY"] = ""
WANDB_SECRET = ""


# ### 2) General Utilities

# In[ ]:


# Utilities
def load_data_from_kaggle(list_of_dbs):
    def load_kaggle_db(full_name, is_competition):
        import kaggle, zipfile
        
        name = full_name
        if not is_competition:
            name = name[name.find("/") + 1:]
        
        if not os.path.isdir(f"../input/{name}"):
            if is_competition:
                kaggle.api.competition_download_files(name)
                zipfile.ZipFile(f"{name}.zip").extractall(f"../input/{name}")
            else:
                kaggle.api.dataset_download_files(full_name, f"../input/{name}", unzip=True)
    
    for db in list_of_dbs:
        load_kaggle_db(*db)

def set_random_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if not IS_KAGGLE and DBS_FROM_KAGGLE is not None:
    load_data_from_kaggle(DBS_FROM_KAGGLE)
set_random_seed(CFG.seed)


# ### 3) Comp utilities

# In[ ]:


### AWP ###

class AWP:
    def __init__(self, model, optimizer, scaler, adv_lr, adv_eps, awp_steps):
        self.model, self.optimizer, self.scaler, self.adv_lr, self.adv_eps, self.awp_steps = model, optimizer, scaler, adv_lr, adv_eps, awp_steps
        self.backup, self.backup_eps = {}, {}
        
        self.is_f16 = CFG.fp16
        print(f"AWP is training with f16 data: {self.is_f16}")
        
    def attack_step(self):
        eps = 1e-6
        for n, p in self.model.named_parameters():
            if p.grad is not None and "weight" in n:
                
                norm1 = torch.norm(p.grad)
                norm2 = torch.norm(p.data.detach())
                
                if norm1 != 0 and not torch.isnan(norm1):
                    
                    r_at = self.adv_lr * p.grad / (norm1 + eps) * (norm2 + eps)
                    
                    p.data.add_(r_at)
                    p.data = torch.min(
                        torch.max(p.data, self.backup_eps[n][0]), self.backup_eps[n][1]
                    )
    
    def attack(self, batch):
        self.save()
        for i in range(self.awp_steps):
            self.attack_step()
            
            with torch.cuda.amp.autocast(enabled=self.is_f16):
                adv_outputs = self.model(**batch)
            
            adv_loss = adv_outputs["loss"]

            self.optimizer.zero_grad()
            self.scaler.scale(adv_loss).backward()

        self.restore()
    
    def save(self):
        for n, p in self.model.named_parameters():
            if p.grad is not None and "weight" in n:
                if n not in self.backup:
                    data               = p.data.clone()#.cpu()
                    self.backup[n]     = data
                    grad_eps           = self.adv_eps * p.abs().detach()
                    self.backup_eps[n] = (data - grad_eps, data + grad_eps)
    
    def restore(self):
        for n, p in self.model.named_parameters():
            if n in self.backup:
                p.data = self.backup[n]#.cuda()
        
        self.backup = {}
        self.backup_eps = {}


### Trainer ###

class AWPModule(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model       = model
        self.best_metric = float("inf")
        
        self.start_awp_epoch = CFG.start_awp_epoch
    
    def on_train_start(self):
        if CFG.use_awp:
            print("START AWP")
            
            scaler = self.trainer.scaler
            if scaler is None:
                scaler = torch.cuda.amp.GradScaler(enabled=False)
                
            self.awp = AWP(model, self.optimizers().optimizer, scaler, CFG.adversarial_lr, CFG.adversarial_eps, CFG.awp_steps)
        else:
            print("NO AWP")
            self.start_awp_epoch = 1000

    def on_after_backward(self):
        if self.current_epoch >= self.start_awp_epoch:
            self.awp.attack(self.batch)
    
    def training_step(self, batch, idx):
        self.batch = batch
        
        outputs = self.model(**batch)
        loss    = outputs["loss"]

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def training_epoch_end(self, outputs):
        loss = torch.tensor([o["loss"] for o in outputs]).mean()
        
        self.log("train/epoch-loss", loss)
        print(f"#### Training {self.current_epoch} epoch mean loss: {loss} ####")
        print("\n")


    def validation_step(self, batch, idx):
        outputs = self.model(**batch)
        outputs["labels"] = batch["labels"]
        return outputs
    
    def validation_epoch_end(self, outputs):
        logits = torch.cat([o["logits"] for o in outputs]).detach().cpu().numpy()
        labels = torch.cat([o["labels"] for o in outputs]).detach().cpu().numpy()
        loss   = torch.tensor([o["loss"] for o in outputs]).mean()
        scores = compute_metrics((logits, labels))

        self.log("val/loss", loss)
        self.log_dict(scores)
        
        saved = False
        if scores["val/mean_score"] < self.best_metric:
            self.best_metric = scores["val/mean_score"]
            saved = True
        string = "Saved" if saved else "Not saved"

        print(f"#### Validation Loss: {loss} -- Mean Score: {scores['val/mean_score']} -- Model {string} ####")
        
    def configure_optimizers(self):
        optimizer, scheduler = optimizers
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

def train(model, optimizers, compute_metrics, eval_steps, train_dataloader, eval_dataloader, train=True):
        
        wrapped_model = AWPModule(model)

        class DataHandler(pl.Callback):
            def on_train_epoch_end(self, trainer, pl_module):
                print("#### Incrementing current epoch count ####")
                trainer.train_dataloader.loaders.dataset.epoch    += 1
                trainer.train_dataloader.loaders.collate_fn.epoch += 1
                                   
#         checkpoint = pl.callbacks.ModelCheckpoint(
#             dirpath           = CFG.save_dir,
#             monitor           = "val/mean_score",
#             mode              = "min",
#             save_weights_only = True,
#             save_last         = True,
#         )

        trainer = pl.Trainer(
            logger                  = pl.loggers.WandbLogger(project=NAME_PROJECT) if CFG.wandb else True,
            callbacks               = [DataHandler(), pl.callbacks.LearningRateMonitor("step")],
            accumulate_grad_batches = CFG.accumulation_steps,
            devices                 = 1,
            accelerator             = "gpu" if CFG.device == "cuda" else "cpu",
            gradient_clip_val       = CFG.max_grad_norm,
            log_every_n_steps       = 5,
            max_epochs              = CFG.max_epochs,
            precision               = 16 if CFG.fp16 else 32,
            enable_progress_bar     = True, 
            val_check_interval      = eval_steps,
        )

        if train:
            trainer.fit(wrapped_model, train_dataloader, eval_dataloader)
        else:
            trainer.validate(model=wrapped_model, dataloaders=eval_dataloader)

### Other utilities ###

def get_cols():
    return [IDS_TO_LABELS[i] for i in range(len(IDS_TO_LABELS))]

def make_folds(df, num_folds, random_state=42):
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
    
    df.loc[:, "fold"] = -1
    split_fn = MultilabelStratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state)
    for n, (train, valid) in enumerate(split_fn.split(df, df[get_cols()])):
        df.loc[valid, "fold"] = n
    return df

def clean_text(text):
    new_text = text.strip()
    return new_text

def reinit_last_layers(transformer_model, num_layers):
    if num_layers > 0:
        transformer_model.encoder.layer[-num_layers:].apply(transformer_model._init_weights)


# ### 4) Prepare Data

# In[ ]:


tokenizer = transformers.AutoTokenizer.from_pretrained(CFG.model_name, use_fast=True)

if CFG.new_line_token:
    tokenizer.add_tokens(["\n"], special_tokens=True)


# In[ ]:


nrows = 20 if CFG.debug else None
df = pd.read_csv("../input/feedback-prize-english-language-learning/train.csv", nrows=nrows)
df = make_folds(df, CFG.num_folds)
df["full_text"] = [clean_text(text) for text in df.full_text]


# In[ ]:


df.head()


# In[ ]:


CFG.mask_prob_0 = 0
CFG.mix_prob_0 = 0


class CDataset(Dataset):
    def __init__(self, df, tokenizer, use_augmentations):
        self.df                = df
        self.tokenizer         = tokenizer
        self.use_augmentations = use_augmentations
        self.epoch             = 0

        print(f"{len(self)} Samples Loaded")

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        items   = self.df.iloc[idx]
        template = "Evaluate the following text on: cohesion syntax vocabulary phraseology grammar conventions"
        text = items["full_text"]
#         encoded = self.tokenizer(template, text, truncation=True, max_length=50)
        encoded = self.tokenizer(template, text)

        # Data Augmentation
        if self.use_augmentations:
            mask_prob = 0.0
            if self.epoch == 0: mask_prob = CFG.mask_prob_0

            # Apply random mask 
            if mask_prob > 0:
                ids            = torch.tensor(encoded["input_ids"])
                probs          = torch.full(ids.shape, mask_prob)
                special_tokens = torch.tensor(self.tokenizer.get_special_tokens_mask(ids, already_has_special_tokens=True))
                probs.masked_fill_(special_tokens, 0.0)

                masked_indices       = torch.bernoulli(probs).bool()
                ids[masked_indices]  = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
                encoded["input_ids"] = ids.tolist()

        encoded["labels"] = items[get_cols()].tolist()
        return encoded

class CCollator:
    def __init__(self, tokenizer, use_augmentations):
        self.tokenizer         = tokenizer
        self.use_augmentations = use_augmentations
        self.epoch             = 0
        
    def __call__(self, batch):
        labels = torch.tensor([item.pop("labels") for item in batch], dtype=torch.float32)
        batch  = self.tokenizer.pad(batch, return_tensors="pt")
        # Data Augmentation
        if self.use_augmentations:
            mix_prob = 0.0
            if self.epoch == 0: mix_prob = CFG.mix_prob_0
            
            if mix_prob > 0:
                if random.random() < mix_prob:
                    ids   = batch["input_ids"]
                    mask  = batch["attention_mask"]
                    types = batch["token_type_ids"]
                    
                    perm  = torch.randperm(ids.shape[0])
                    rand_len = int(ids.shape[1] * 0.25)
                    start = torch.randint(15, ids.shape[1] - rand_len, (1,)) # IMPORTANT: Hard coded min (len of template)

                    ids[:, start:start+rand_len] = ids[perm, start:start+rand_len]
                    mask[:, start:start+rand_len] = mask[perm, start:start+rand_len]
                    types[:, start:start+rand_len] = types[perm, start:start+rand_len]
                    
                    
        batch["labels"] = labels
        return batch

def one_batch():
    ds       = CDataset(df, tokenizer, True)
    collator = CCollator(tokenizer, True)
    dl       = DataLoader(ds, batch_size=2, collate_fn=collator)
    return next(iter(dl))


# ### 5) Model, optimizer, loss and metrics

# In[ ]:


LOSS_FN = nn.MSELoss
# LOSS_FN = nn.SmoothL1Loss
# LOSS_FN = nn.L1Loss

def compute_metrics(eval_preds):
    # Mean Columnwise Root Mean Squarred Error
    ids, labels = eval_preds
    scores      = []
    idxes       = labels.shape[1]

    for i in range(idxes):
        pred  = ids[:, i]
        label = labels[:, i]
        score = mean_squared_error(label, pred, squared=False) # RMSE
        scores.append(score)

    mean_score = np.mean(scores)
    metrics = {"val/mean_score": mean_score, **{f"val/{v}": scores[k] for k,v in IDS_TO_LABELS.items()}}
    return metrics


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

class AttentionPool(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionPool, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
    
    def forward(self, hidden_state, attention_mask):
        w = self.attention(hidden_state)
        w[attention_mask == 0] = float("-inf")
        w = torch.softmax(w, dim=1)
        context = torch.sum(hidden_state * w, dim=1)
        return context

    
class LinearAttention(nn.Module):
    def __init__(self, hidden_size):
        super(LinearAttention, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, hidden_state, attention_mask):
        w = self.attention(hidden_state)
        w[attention_mask == 0] = float("-inf")
        w = torch.softmax(w, dim=1)

        context = torch.sum(hidden_state * w, dim=1)
        return context


class MultiSampleHead(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super(MultiSampleHead, self).__init__()
        self.drops   = nn.ModuleList([nn.Dropout(0.1 * i) for i in range(1,6)])
        self.cls     = nn.Linear(hidden_size, num_labels)
        self.loss_fn = LOSS_FN()

    def forward(self, hidden_state, labels=None):
        loss   = 0
        logits = torch.stack([
            self.cls(drop(hidden_state)) for drop in self.drops
        ]).mean(dim=0)

        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {
            "logits" : logits,
            "loss"   : loss
        }
    

class SimpleHead(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super(SimpleHead, self).__init__()
        self.cls     = nn.Linear(hidden_size, num_labels)
        self.loss_fn = LOSS_FN()
    
    def forward(self, hidden_state, labels=None):
        loss         = 0
        hidden_state = self.cls(hidden_state)
        
        if labels is not None:
            loss = self.loss_fn(hidden_state, labels)
        
        return {
            "logits": hidden_state,
            "loss"  : loss
        }


class LSTMBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=1, p_drop=0):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_channels,
                             hidden_size=out_channels,
                             num_layers=num_layers,
                             dropout=p_drop,
                             batch_first=True, 
                             bidirectional=True)
    def forward(self, x):
        x,_ = self.lstm(x)
        return x

class CNet(nn.Module):
    def __init__(self, model_name, num_labels, embed_size, pretrained=True):
        super(CNet, self).__init__()
        config = transformers.AutoConfig.from_pretrained(model_name)
        config.update({
            "hidden_dropout_prob"         : 0.0,
            "attention_probs_dropout_prob": 0.0,
            "add_pooling_layer"           : False,
            "num_labels"                  : num_labels,
            "rnn_dropout"                 : CFG.rnn_dropout,
            "head_dropout"                : CFG.head_dropout
        })
        
        self.config = config
        
        if pretrained:
            self.transformer = transformers.AutoModel.from_pretrained(model_name, config=config)
            self.transformer.resize_token_embeddings(embed_size)
        else:
            self.transformer = transformers.AutoModel.from_config(config)
        
        hidden_size = config.hidden_size * 4
        self.lstm = LSTMBlock(hidden_size, hidden_size // 2, p_drop=CFG.rnn_dropout)

        self.pool = AttentionPool(hidden_size)
        
        self.head_drop = nn.Dropout(CFG.head_dropout)
        self.head = SimpleHead(hidden_size, num_labels)
        
        if CFG.gradient_checkpoint:
            self.transformer.gradient_checkpointing_enable()
        
        self._init_weights(self.head)
        
        reinit_last_layers(self.transformer, 1)
        
    def _init_weights(self, module):
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

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        out = self.transformer(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True).hidden_states
        
        out = torch.cat(out[-4:], dim=-1)
        
        out = self.lstm(out)
        
        out = self.pool(out, attention_mask)

        out = self.head_drop(out)

        out = self.head(out, labels)
        
        return out


# In[ ]:


def get_optimizer(CFG, model, total_steps, warm_steps):

    all_parameters = list(model.named_parameters())
    used_name_parameters = set()

    params = []

    no_wd = ["word_embeddings", "bias", "LayerNorm.weight"]

    # Head parameters
    head =  [(name, param) for name, param in all_parameters if not "transformer" in name]
    for n,_ in head: used_name_parameters.add(n)
    
    params.append({"params": [p for n,p in head if not any(nd in n for nd in no_wd)], "weight_decay": 0.01, "lr": CFG.head_lr})
    params.append({"params": [p for n,p in head if any(nd in n for nd in no_wd)], "weight_decay": 0., "lr": CFG.head_lr})

    # Backend parameters
    transformer = [(name, param) for name, param in all_parameters if "transformer" in name]
    groups = [
        [ [".embeddings."] ,                           1e-6],
        [ ["encoder.LayerNorm", "rel_embeddings"],     1e-6],
        [ ["." + str(i) + "." for i in range(0,6)],    1e-8],
        [ ["." + str(i) + "." for i in range(6,12)],   1e-7],
        [ ["." + str(i) + "." for i in range(12,23)],  CFG.transformer_lr],
        [ [".23."] , CFG.head_lr ]
    ]
    
    for group in groups:
        grouped_names = group[0]
        lr = group[1]

        parameters = [(name, param) for name,param in transformer if any(gn in name for gn in grouped_names)]
        for n, _ in parameters: used_name_parameters.add(n)
        
        params.append({"params": [p for n,p in parameters if not any(nd in n for nd in no_wd)], "weight_decay": 0.01, "lr": lr})
        params.append({"params": [p for n,p in parameters if any(nd in n for nd in no_wd)], "weight_decay": 0., "lr": lr})

    state_dict_keys = {n:p for n,p in all_parameters}.keys()
    assert len(state_dict_keys - used_name_parameters) == 0,    f"Missing parameters: {str(state_dict_keys - used_name_parameters)}"
    
    if not CFG.optimizer_bit8:
        optimizer = torch.optim.AdamW(params, eps=CFG.adam_eps)
    elif CFG.device == "cuda":
        optimizer = bnb.optim.AdamW8bit(params, eps=CFG.adam_eps)
    scheduler = transformers.get_scheduler(CFG.scheduler, optimizer, warm_steps, total_steps)

    return (optimizer, scheduler)


# ### 6) Train!!

# In[ ]:


# # Init wandb
CFG.wandb = False
if not CFG.debug and len(WANDB_SECRET) > 0:
    WANDB = wandb.login(key=WANDB_SECRET)

    if WANDB:
        wandb.init(
            project=NAME_PROJECT,
            config={k:v for k,v in CFG.__dict__.items() if not k.startswith("_")},
            save_code=True
        )

    CFG.wandb = WANDB


# In[ ]:


for key, value in CFG.__dict__.items():
    if not key.startswith("_"):
        print(f"{key}: {value}")


# In[ ]:


model = CNet(CFG.model_name, len(IDS_TO_LABELS), len(tokenizer), True)


# In[ ]:


train_df = df[df.fold != CFG.current_fold]
valid_df = df[df.fold == CFG.current_fold]

valid_df.loc[:, "lens"] = [len(text) for text in valid_df.full_text]
valid_df = valid_df.sort_values(by="lens", ascending=False)

train_ds = CDataset(train_df, tokenizer, CFG.use_augmentation)
train_collator = CCollator(tokenizer, CFG.use_augmentation)

valid_ds = CDataset(valid_df, tokenizer, False)
valid_collator = CCollator(tokenizer, False)

train_dataloader = DataLoader(train_ds, CFG.batch_size, shuffle=True, collate_fn=train_collator, num_workers=CFG.num_jobs, pin_memory=True)
valid_dataloader = DataLoader(valid_ds, CFG.eval_batch_size, shuffle=False, collate_fn=valid_collator, num_workers=CFG.num_jobs, pin_memory=True)


total_steps = int(len(train_dataloader) / CFG.accumulation_steps * CFG.max_epochs)

if isinstance(CFG.warmup_epochs, float):
    warm_steps = int(total_steps * CFG.warmup_epochs)
else:
    warm_steps = int(len(train_dataloader) * CFG.warmup_epochs)

optimizers = get_optimizer(CFG, model, total_steps, warm_steps)

total_parameters = sum([param.data.nelement() for param in model.parameters()])

print(f"Using device: {CFG.device_name}")
print(f"Training for a total of {total_steps} steps")
print(f"Warming for {warm_steps} steps")
print(f"Total parameters to optimize {total_parameters}")
print(LOSS_FN)


# In[ ]:


if CFG.device == "cuda":
    train(model, optimizers, compute_metrics, 1., train_dataloader, valid_dataloader)

