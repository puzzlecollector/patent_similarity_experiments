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

### ignore warning
def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)
            
set_global_logging_level(logging.ERROR, ["transformers", "nlp", "torch", "tensorflow", "tensorboard", "wandb"])


class TripletData(Dataset):
    def __init__(self, path):
        super(TripletData, self).__init__()
        self.data = [txt for txt in Path(path).glob('*.txt')]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
    
class custom_collate(object):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("tanapatentlm/patentdeberta_base_spec_1024_pwi")
        self.seq_len = 10
        self.chunk_size = 128 
        
    def chunk_tokens(self, tokens, start_token_id, end_token_id, overlap=10, chunk_size=128):
        chunk_size = chunk_size - self.tokenizer.num_special_tokens_to_add()
        total, partial = [], []
        if len(tokens) / (chunk_size - overlap) > 0:
            n = math.ceil(len(tokens) / (chunk_size - overlap))
        else:
            n = 1
        for w in range(n):
            if w == 0:
                partial = tokens[:chunk_size]
            else:
                partial = tokens[w * (chunk_size - overlap):w * (chunk_size - overlap) + chunk_size]
            partial = [start_token_id] + partial + [end_token_id]
            total.append(partial)
        return total
    
    def get_chunked(self, t, seq_len = 10, window_size = 128):
        encoded_inputs = self.tokenizer(t, add_special_tokens=False)
        input_ids = encoded_inputs['input_ids']
        attention_masks = encoded_inputs['attention_mask']
        chunks = self.chunk_tokens(input_ids, self.tokenizer.cls_token_id, self.tokenizer.sep_token_id)
        chunk_attention_masks = self.chunk_tokens(attention_masks, 1, 1)
        if len(chunks) > seq_len:
            chunks = chunks[:seq_len]
            chunk_attention_masks = chunk_attention_masks[:seq_len]
        else:
            while len(chunks) < seq_len:
                chunks.append([])
                chunk_attention_masks.append([])
        for i in range(len(chunks)):
            while len(chunks[i]) < window_size:
                chunks[i].append(0)
                chunk_attention_masks[i].append(0)
        chunks = torch.tensor(chunks, dtype=int)
        chunk_attention_masks = torch.tensor(chunk_attention_masks, dtype=int)
        return chunks, chunk_attention_masks
    
    def __call__(self, batch):
        b = len(batch) 
        qb_input_ids, qb_attn_masks = torch.zeros((b, self.seq_len, self.chunk_size),dtype=int), torch.zeros((b, self.seq_len, self.chunk_size),dtype=int)
        pb_input_ids, pb_attn_masks = torch.zeros((b, self.seq_len, self.chunk_size),dtype=int), torch.zeros((b, self.seq_len, self.chunk_size),dtype=int)
        nb_input_ids, nb_attn_masks = torch.zeros((b, self.seq_len, self.chunk_size),dtype=int), torch.zeros((b, self.seq_len, self.chunk_size),dtype=int)
        labels = torch.ones((b), dtype=int)
        for idx, txt_file in enumerate(batch):
            with txt_file.open("r", encoding="utf8") as f:
                data = f.read()
            triplet = data.split("\n\n\n") 
            q, p, n = triplet
            q_input_ids, q_attn_masks = self.get_chunked(q)
            p_input_ids, p_attn_masks = self.get_chunked(p)
            n_input_ids, n_attn_masks = self.get_chunked(n)
            qb_input_ids[idx] = q_input_ids
            qb_attn_masks[idx] = q_attn_masks
            pb_input_ids[idx] = p_input_ids 
            pb_attn_masks[idx] = p_attn_masks 
            nb_input_ids[idx] = n_input_ids
            nb_attn_masks[idx] = n_attn_masks 
        return qb_input_ids, qb_attn_masks, pb_input_ids, pb_attn_masks, nb_input_ids, nb_attn_masks, labels
    
train_set = TripletData("../storage/patent_experiments/FGH_claim_triplet_v0.1s/train")
valid_set = TripletData("../storage/patent_experiments/FGH_claim_triplet_v0.1s/valid") 
collate = custom_collate()
train_dataloader = DataLoader(train_set, batch_size=12, num_workers=4, collate_fn=collate, shuffle=True)
validation_dataloader = DataLoader(valid_set, batch_size=12, num_workers=4, collate_fn=collate, shuffle=False) 


### define model ###     
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000): 
        super().__init__() 
        self.dropout = nn.Dropout(p=dropout) 
        position = torch.arange(max_len).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.pe = torch.zeros(1, max_len, d_model) 
        self.pe[0, :, 0::2] = torch.sin(position * div_term) 
        self.pe[0, :, 1::2] = torch.cos(position * div_term)  
        self.device = torch.device('cuda') 
    def forward(self, x:Tensor) -> Tensor: 
        ''' input: [batch_size, seq_len, hidden_dim] ''' 
        self.pe = self.pe.to(self.device) 
        x = x + self.pe[:,:x.size(1),:] 
        return self.dropout(x) 
    
class DeBERTa_Ranker(torch.nn.Module):
    def __init__(self): 
        super(DeBERTa_Ranker, self).__init__() 
        self.seq_len = 10
        self.window_size = 128 
        self.d_model = 768 
        self.device = torch.device('cuda') 
        self.tokenizer = AutoTokenizer.from_pretrained("tanapatentlm/patentdeberta_base_spec_1024_pwi")
        self.chunk_LM = AutoModel.from_pretrained("tanapatentlm/patentdeberta_base_spec_1024_pwi")
        self.pos_encoder = PositionalEncoding(d_model=self.d_model) 
        encoder_layers = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=8, batch_first=True) 
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=4)
        
    def forward(self, input_ids, attention_mask):
        batch = input_ids.size(0) 
        input_ids = input_ids.view(-1, self.window_size) 
        attention_mask = attention_mask.view(-1, self.window_size) 
        x = self.chunk_LM(input_ids=input_ids, attention_mask=attention_mask) 
        x = x[0][:,0,:] 
        x = x.view(batch, self.seq_len, self.d_model)  
        cls_tokens = self.tokenizer([self.tokenizer.cls_token for _ in range(batch)], return_tensors='pt', add_special_tokens=False).to(self.device) 
        cls_embs = self.chunk_LM(**cls_tokens)[0]        
        x = torch.cat((cls_embs, x), dim=1)         
        x = self.pos_encoder(x) 
        x = self.transformer_encoder(x) 
        x = x[:,0,:] 
        return x 

### train model ### 
model = DeBERTa_Ranker()
model.cuda()

accumulation_steps = 10

# loss_func = nn.TripletMarginWithDistanceLoss(distance_function = lambda x, y: 1.0 - F.cosine_similarity(x,y))  

dist_func = nn.PairwiseDistance(p=2) 
triplet_loss = nn.TripletMarginLoss() 
ce_loss = nn.CrossEntropyLoss() 
optimizer = AdamW(model.parameters(), lr=2e-5) 
epochs = 2
total_steps = len(train_dataloader) * epochs // accumulation_steps  
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=int(0.1*total_steps), 
                                            num_training_steps=total_steps) 


train_losses, val_losses = [], []
device = torch.device('cuda') 
model.zero_grad() 


for epoch_i in tqdm(range(epochs), desc="Epochs", position=0, leave=True, total=epochs):
    train_loss = 0.0 
    train_dist_diff = 0.0 
    model.train()
    iters = 0 # keeps track of number of model updates 
    with tqdm(train_dataloader, unit="batch") as tepoch:
        for step, batch in enumerate(tepoch):
            batch = tuple(t.to(device) for t in batch) 
            q_input_ids, q_input_mask, p_input_ids, p_input_mask, n_input_ids, n_input_mask, b_labels = batch 
            q_emb = model(q_input_ids, q_input_mask) 
            p_emb = model(p_input_ids, p_input_mask) 
            n_emb = model(n_input_ids, n_input_mask) 
            
            p_score = dist_func(q_emb, p_emb).to(device)
            n_score = dist_func(q_emb, n_emb).to(device)  
            p_score = p_score.view(p_score.size(0),-1) 
            n_score = n_score.view(n_score.size(0),-1) 
            dist_scores = torch.cat((p_score, n_score), dim=1) 
            
            p_dist_sum = torch.sum(dist_scores[:,0])  
            n_dist_sum = torch.sum(dist_scores[:,1])
            dist_diff = p_dist_sum - n_dist_sum # better for this to be minimized: the more negative -> better embedding  
            train_dist_diff += dist_diff.item()          
            loss = triplet_loss(q_emb, p_emb, n_emb)
            
            loss = loss / accumulation_steps 
            loss.backward() 
            if (step+1)%accumulation_steps == 0 or (step + 1 == len(train_dataloader)):
                train_loss += loss.item() * accumulation_steps 
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
                optimizer.step() 
                scheduler.step() 
                model.zero_grad() 
                tepoch.set_postfix(loss=train_loss/(iters+1), diff=train_dist_diff/(iters+1)) 
                iters += 1 
                time.sleep(0.1)  
    avg_train_loss = train_loss / iters 
    print("average train loss : {}".format(avg_train_loss)) 
    train_losses.append(avg_train_loss) 
    
    val_loss = 0.0 
    model.eval() 
    for step, batch in tqdm(enumerate(validation_dataloader), desc="Validating", position=0, leave=True, total=len(validation_dataloader)):
        batch = tuple(t.to(device) for t in batch) 
        q_input_ids, q_input_mask, p_input_ids, p_input_mask, n_input_ids, n_input_mask, b_labels = batch 
        with torch.no_grad():
            q_emb = model(q_input_ids, q_input_mask) 
            p_emb = model(p_input_ids, p_input_mask) 
            n_emb = model(n_input_ids, n_input_mask)

        p_score = dist_func(q_emb, p_emb).to(device)
        n_score = dist_func(q_emb, n_emb).to(device)  
        p_score = p_score.view(p_score.size(0),-1) 
        n_score = n_score.view(n_score.size(0),-1) 
        dist_scores = torch.cat((p_score, n_score), dim=1) 
                    
        loss = triplet_loss(q_emb, p_emb, n_emb) 

        val_loss += loss.item() 
    avg_val_loss = val_loss / len(validation_dataloader) 
    print("average validation loss: {}".format(avg_val_loss)) 
    val_losses.append(avg_val_loss) 
    
    print("saving checkpoint") 
    torch.save(model.state_dict(), "../storage/PARADE_epoch_{}_train_loss_{}_val_loss_{}.pt".format(epoch_i+1, avg_train_loss, avg_val_loss)) 
