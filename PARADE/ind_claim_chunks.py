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

os.environ["TOKENIZERS_PARALLELISM"] = "false" 

### ignore warning ### 
def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)
            
set_global_logging_level(logging.ERROR, ["transformers", "nlp", "torch", "tensorflow", "tensorboard", "wandb"])

### define dataloader ### 

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
        self.seq_len = 6
        self.chunk_size = 256 
    
    def clean_text(self, t):
        x = re.sub("\d+.","", t) 
        x = x.replace("\n"," ") 
        x = x.strip() 
        return x 
    
    def __call__(self, batch):
        b = len(batch) 
        qb_input_ids, qb_attn_masks = torch.zeros((b, self.seq_len, self.chunk_size),dtype=int), torch.zeros((b, self.seq_len, self.chunk_size),dtype=int)
        pb_input_ids, pb_attn_masks = torch.zeros((b, self.seq_len, self.chunk_size),dtype=int), torch.zeros((b, self.seq_len, self.chunk_size),dtype=int)
        nb_input_ids, nb_attn_masks = torch.zeros((b, self.seq_len, self.chunk_size),dtype=int), torch.zeros((b, self.seq_len, self.chunk_size),dtype=int)
        for idx, txt_file in enumerate(batch):
            with txt_file.open("r", encoding="utf8") as f:
                data = f.read()
            triplet = data.split("\n\n\n") 
            q, p, n = triplet
            q_splitted = q.split('\n\n') 
            p_splitted = p.split('\n\n') 
            n_splitted = n.split('\n\n')
            
            for j in range(min(len(q_splitted), self.seq_len)):
                cleaned_claim = self.clean_text(q_splitted[j]) 
                encoded_inputs = self.tokenizer(cleaned_claim, return_tensors='pt', max_length=self.chunk_size, padding='max_length', truncation=True) 
                qb_input_ids[idx, j, :] = encoded_inputs['input_ids'] 
                qb_attn_masks[idx, j, :] = encoded_inputs['attention_mask'] 
            for j in range(min(len(p_splitted), self.seq_len)):
                cleaned_claim = self.clean_text(p_splitted[j]) 
                encoded_inputs = self.tokenizer(cleaned_claim, return_tensors='pt', max_length=self.chunk_size, padding='max_length', truncation=True)
                pb_input_ids[idx, j, :] = encoded_inputs['input_ids'] 
                pb_attn_masks[idx, j, :] = encoded_inputs['attention_mask'] 
            for j in range(min(len(n_splitted), self.seq_len)):
                cleaned_claim = self.clean_text(n_splitted[j])
                encoded_inputs = self.tokenizer(cleaned_claim, return_tensors='pt', max_length=self.chunk_size, padding='max_length', truncation=True) 
                nb_input_ids[idx, j, :] = encoded_inputs['input_ids'] 
                nb_attn_masks[idx, j, :] = encoded_inputs['attention_mask']     
        return qb_input_ids, qb_attn_masks, pb_input_ids, pb_attn_masks, nb_input_ids, nb_attn_masks
    
train_set = TripletData("../storage/patent_experiments/FGH_claim_triplet_v0.1s/train")
valid_set = TripletData("../storage/patent_experiments/FGH_claim_triplet_v0.1s/valid") 
collate = custom_collate()
train_dataloader = DataLoader(train_set, batch_size=6, num_workers=4, collate_fn=collate, shuffle=True)
validation_dataloader = DataLoader(valid_set, batch_size=6, num_workers=4, collate_fn=collate, shuffle=False) 

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
    def __init__(self, continual_learning=False): 
        super(DeBERTa_Ranker, self).__init__() 
        self.seq_len = 3
        self.window_size = 512 
        self.d_model = 768 
        self.device = torch.device('cuda') 
        self.tokenizer = AutoTokenizer.from_pretrained("tanapatentlm/patentdeberta_base_spec_1024_pwi")
        self.chunk_LM = AutoModel.from_pretrained("tanapatentlm/patentdeberta_base_spec_1024_pwi")
        self.continual_learning = continual_learning 
        if self.continual_learning == True:
            print("loading from weak FirstP checkpoint...")
            ckpt_path = "../storage/checkpoints/deberta_base_spec_1024_pwi/checkpoints-epoch=08-val_loss=0.06.ckpt" 
            checkpoint = torch.load(ckpt_path)
            new_weights = self.chunk_LM.state_dict() 
            old_weights = list(checkpoint['state_dict'].items()) 
            i=0
            for k, _ in new_weights.items():
                new_weights[k] = old_weights[i][1]
                i += 1
            self.chunk_LM.load_state_dict(new_weights) 

        self.pos_encoder = PositionalEncoding(d_model=self.d_model) 
        encoder_layers = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=12, batch_first=True) 
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
model = DeBERTa_Ranker(continual_learning=True)
print("loading from previously saved intermediate checkpoint...") 
checkpoint = torch.load("../storage/intermediate_checkpoint_epoch_1.pt") 
model.load_state_dict(checkpoint) 
model.cuda()

accumulation_steps = 20

loss_func = nn.TripletMarginLoss() 
optimizer = AdamW(model.parameters(), lr=1e-4) 
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
    model.train()
    iters = 0 # keeps track of number of model updates based on the accumulation steps 
    with tqdm(train_dataloader, unit="batch") as tepoch:
        for step, batch in enumerate(tepoch):
            if step%30000==0 and step > 0:
                print("saving intermediate checkpoint") 
                torch.save(model.state_dict(), "../storage/intermediate_checkpoint_epoch_{}_steps_{}.pt".format(epoch_i+1, step)) 
            batch = tuple(t.to(device) for t in batch) 
            q_input_ids, q_input_mask, p_input_ids, p_input_mask, n_input_ids, n_input_mask = batch 
            q_emb = model(q_input_ids, q_input_mask) 
            p_emb = model(p_input_ids, p_input_mask) 
            n_emb = model(n_input_ids, n_input_mask) 
            loss = loss_func(q_emb, p_emb, n_emb)  
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
        loss = loss_func(q_emb, p_emb, n_emb) 
        val_loss += loss.item() 
    avg_val_loss = val_loss / len(validation_dataloader) 
    print("average validation loss: {}".format(avg_val_loss)) 
    val_losses.append(avg_val_loss) 
    
    print("saving checkpoint")  
    torch.save(model.state_dict(), "../storage/IND_CHUNKS_PARADE_epoch_{}_train_loss_{}_val_loss_{}.pt".format(epoch_i+1, avg_train_loss, avg_val_loss)) 
