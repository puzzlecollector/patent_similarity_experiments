import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from torch import Tensor
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler, IterableDataset
import math
import time
import datetime
import os
import re
from sklearn.model_selection import train_test_split, StratifiedKFold
import logging
import addict
from pathlib import Path
from pytorch_metric_learning import miners, losses 


os.environ["TOKENIZERS_PARALLELISM"] = "false" 

### ignore warning ### 
def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)
            
set_global_logging_level(logging.ERROR, ["transformers", "nlp", "torch", "tensorflow", "tensorboard", "wandb"])


tokenizer = AutoTokenizer.from_pretrained("google/bigbird-pegasus-large-bigpatent") 

### define dataloader ### 
class TripletData(Dataset):
    def __init__(self, path):
        super(TripletData, self).__init__()
        self.data = [txt for txt in Path(path).glob('*.txt')]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
    
class custom_collate_metric(object):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("google/bigbird-pegasus-large-bigpatent")
        self.chunk_size = 1024 
    
    def clean_text(self, t):
        x = re.sub("\d+.","", t) 
        x = x.replace("\n"," ") 
        x = x.strip() 
        return x 

    def __call__(self, batch):
        b = len(batch) 
        input_ids, attn_masks = [], [] 
        labels = [] 
        ids = 0         
        for idx, txt_file in enumerate(batch):
            with txt_file.open("r", encoding="utf8") as f:
                data = f.read() 
            triplet = data.split("\n\n\n") 
            q, p, n = triplet
            q_claims = q.split('\n\n') 
            p_claims = p.split('\n\n') 
            n_claims = n.split('\n\n')
            
            q_input_ids, q_attn_masks = torch.zeros((self.chunk_size),dtype=int), torch.zeros((self.chunk_size),dtype=int)
            p_input_ids, p_attn_masks = torch.zeros((self.chunk_size),dtype=int), torch.zeros((self.chunk_size),dtype=int)
            n_input_ids, n_attn_masks = torch.zeros((self.chunk_size),dtype=int), torch.zeros((self.chunk_size),dtype=int) 
            
            # combine text based on claim numbers separated by SEP token  
            q_clean_claims = [] 
            for q_claim in q_claims:
                q_clean = self.clean_text(q_claim) 
                q_clean_claims.append(q_clean) 
            q_text = ""
            for i in range(len(q_clean_claims)):
                if i == len(q_clean_claims)-1:
                    q_text += str(q_clean_claims[i]) 
                else:
                    q_text += str(q_clean_claims[i]) + self.tokenizer.sep_token 
            
            p_clean_claims = [] 
            for p_claim in p_claims:
                p_clean = self.clean_text(p_claim) 
                p_clean_claims.append(p_clean) 
            p_text = ""
            for i in range(len(p_clean_claims)):
                if i == len(p_clean_claims)-1:
                    p_text += str(p_clean_claims[i]) 
                else:
                    p_text += str(p_clean_claims[i]) + self.tokenizer.sep_token
            
            n_clean_claims = [] 
            for n_claim in n_claims:
                n_clean = self.clean_text(n_claim) 
                n_clean_claims.append(n_clean) 
            n_text = ""
            for i in range(len(n_clean_claims)):
                if i == len(n_clean_claims)-1:
                    n_text += str(n_clean_claims[i]) 
                else:
                    n_text += str(n_clean_claims[i]) + self.tokenizer.sep_token
                    
            encoded_q = self.tokenizer(q_text, return_tensors='pt', max_length=self.chunk_size, padding='max_length', truncation=True)
            encoded_p = self.tokenizer(p_text, return_tensors='pt', max_length=self.chunk_size, padding='max_length', truncation=True) 
            encoded_n = self.tokenizer(n_text, return_tensors='pt', max_length=self.chunk_size, padding='max_length', truncation=True) 
                
            input_ids.append(encoded_q['input_ids'])
            attn_masks.append(encoded_q['attention_mask']) 
            labels.append(ids*2) 
            
            input_ids.append(encoded_p['input_ids'])
            attn_masks.append(encoded_p['attention_mask']) 
            labels.append(ids*2) 
            
            input_ids.append(encoded_n['input_ids'])
            attn_masks.append(encoded_n['attention_mask']) 
            labels.append(ids*2 + 1) 
            ids += 1 
            
        input_ids = torch.stack(input_ids, dim=0).squeeze(dim=1)
        attn_masks = torch.stack(attn_masks, dim=0).squeeze(dim=1)    
        labels = torch.tensor(labels, dtype=int) 
        return input_ids, attn_masks, labels
    
class custom_collate(object):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("google/bigbird-pegasus-large-bigpatent")
        self.chunk_size = 1024 
    
    def clean_text(self, t):
        x = re.sub("\d+.","", t) 
        x = x.replace("\n"," ") 
        x = x.strip() 
        return x 
    
    def __call__(self, batch):
        b = len(batch) 
        qb_input_ids, qb_attn_masks = torch.zeros((b, self.chunk_size),dtype=int), torch.zeros((b, self.chunk_size),dtype=int)
        pb_input_ids, pb_attn_masks = torch.zeros((b, self.chunk_size),dtype=int), torch.zeros((b, self.chunk_size),dtype=int)
        nb_input_ids, nb_attn_masks = torch.zeros((b, self.chunk_size),dtype=int), torch.zeros((b, self.chunk_size),dtype=int)
        
        for idx, txt_file in enumerate(batch):
            with txt_file.open("r", encoding="utf8") as f:
                data = f.read()
            triplet = data.split("\n\n\n") 
            q, p, n = triplet
            q_claims = q.split('\n\n') 
            p_claims = p.split('\n\n') 
            n_claims = n.split('\n\n') 
            
            # combine text based on claim numbers separated by SEP token  
            q_clean_claims = [] 
            for q_claim in q_claims:
                q_clean = self.clean_text(q_claim) 
                q_clean_claims.append(q_clean) 
            q_text = ""
            for i in range(len(q_clean_claims)):
                if i == len(q_clean_claims)-1:
                    q_text += str(q_clean_claims[i]) 
                else:
                    q_text += str(q_clean_claims[i]) + self.tokenizer.sep_token 
            
            p_clean_claims = [] 
            for p_claim in p_claims:
                p_clean = self.clean_text(p_claim) 
                p_clean_claims.append(p_clean) 
            p_text = ""
            for i in range(len(p_clean_claims)):
                if i == len(p_clean_claims)-1:
                    p_text += str(p_clean_claims[i]) 
                else:
                    p_text += str(p_clean_claims[i]) + self.tokenizer.sep_token
            
            n_clean_claims = [] 
            for n_claim in n_claims:
                n_clean = self.clean_text(n_claim) 
                n_clean_claims.append(n_clean) 
            n_text = ""
            for i in range(len(n_clean_claims)):
                if i == len(n_clean_claims)-1:
                    n_text += str(n_clean_claims[i]) 
                else:
                    n_text += str(n_clean_claims[i]) + self.tokenizer.sep_token
            
            encoded_q = self.tokenizer(q_text, return_tensors='pt', max_length=self.chunk_size, padding='max_length', truncation=True)
            encoded_p = self.tokenizer(p_text, return_tensors='pt', max_length=self.chunk_size, padding='max_length', truncation=True) 
            encoded_n = self.tokenizer(n_text, return_tensors='pt', max_length=self.chunk_size, padding='max_length', truncation=True) 
            qb_input_ids[idx] = encoded_q['input_ids'] 
            qb_attn_masks[idx] = encoded_q['attention_mask'] 
            pb_input_ids[idx] = encoded_p['input_ids'] 
            pb_attn_masks[idx] = encoded_p['attention_mask'] 
            nb_input_ids[idx] = encoded_n['input_ids'] 
            nb_attn_masks[idx] = encoded_n['attention_mask'] 
        return qb_input_ids, qb_attn_masks, pb_input_ids, pb_attn_masks, nb_input_ids, nb_attn_masks
    
    
train_set = TripletData("../storage/patent_experiments/FGH_claim_triplet_v0.1s/train")
valid_set = TripletData("../storage/patent_experiments/FGH_claim_triplet_v0.1s/valid") 
collate = custom_collate()
collate_train = custom_collate_metric() 
train_dataloader = DataLoader(train_set, batch_size=2, num_workers=4, collate_fn=collate_train, shuffle=True)
validation_dataloader = DataLoader(valid_set, batch_size=2, num_workers=4, collate_fn=collate, shuffle=False) 

class BigBirdRanker(torch.nn.Module):
    def __init__(self): 
        super(BigBirdRanker, self).__init__() 
        self.d_model = 1024 
        self.device = torch.device('cuda') 
        self.model = AutoModel.from_pretrained("google/bigbird-pegasus-large-bigpatent")
        
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float() 
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
    def forward(self, input_ids, attention_mask):
        x = self.model(input_ids, attention_mask) 
        return self.mean_pooling(x, attention_mask) 

train_losses, val_losses = [],[] 
accumulation_steps = 4

model = BigBirdRanker() 
model.cuda() 

miner = miners.TripletMarginMiner() 
loss_func = losses.TripletMarginLoss() 
torch_loss_func = nn.TripletMarginLoss() 

optimizer = AdamW(model.parameters(), lr=1e-4) 
epochs = 3
device = torch.device('cuda') 
total_steps = len(train_dataloader) * epochs // accumulation_steps  
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=int(0.1*total_steps), 
                                            num_training_steps=total_steps) 


device = torch.device('cuda') 
model.zero_grad() 


for epoch_i in tqdm(range(epochs), desc="Epochs", position=0, leave=True, total=epochs):
    train_loss = 0.0 
    model.train()
    iters = 0 # keeps track of number of model updates based on the accumulation steps 
    with tqdm(train_dataloader, unit="batch") as tepoch:
        for step, batch in enumerate(tepoch):
            if step%1000==0 and step > 0:
                print("saving intermediate checkpoint") 
                torch.save(model.state_dict(), "../storage/BIGBIRD_intermediate_checkpoint_epoch_{}_steps_{}.pt".format(epoch_i+1, step)) 

            batch = tuple(t.to(device) for t in batch) 
            input_ids, attn_masks, labels = batch
            
            
            embeddings = model(input_ids, attn_masks) 
            # hard_pairs = miner(embeddings, labels) 
            #loss = loss_func(embeddings, labels, hard_pairs) 
            loss = loss_func(embeddings, labels)  
            loss = loss / accumulation_steps 
            loss.backward() 
            if (step+1)%accumulation_steps == 0 or (step + 1 == len(train_dataloader)):
                train_loss += loss.item() * accumulation_steps 
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
                optimizer.step() 
                scheduler.step() 
                model.zero_grad() 
                tepoch.set_postfix(loss=train_loss/(iters+1)) 
                iters += 1 
                time.sleep(0.1)  
    avg_train_loss = train_loss / iters 
    print("average train loss : {}".format(avg_train_loss)) 
    train_losses.append(avg_train_loss) 
    
    val_loss = 0.0 
    model.eval() 
    for step, batch in tqdm(enumerate(validation_dataloader), desc="Validating", position=0, leave=True, total=len(validation_dataloader)):
        batch = tuple(t.to(device) for t in batch) 
        q_input_ids, q_input_mask, p_input_ids, p_input_mask, n_input_ids, n_input_mask = batch 
        with torch.no_grad():
            q_emb = model(q_input_ids, q_input_mask) 
            p_emb = model(p_input_ids, p_input_mask) 
            n_emb = model(n_input_ids, n_input_mask)
        loss = torch_loss_func(q_emb, p_emb, n_emb) 
        val_loss += loss.item() 
    avg_val_loss = val_loss / len(validation_dataloader) 
    print("average validation loss: {}".format(avg_val_loss)) 
    val_losses.append(avg_val_loss) 
    
    print("saving checkpoint")  
    torch.save(model.state_dict(), "../storage/BIGBIRD_epoch_{}_train_loss_{}_val_loss_{}.pt".format(epoch_i+1, avg_train_loss, avg_val_loss)) 
