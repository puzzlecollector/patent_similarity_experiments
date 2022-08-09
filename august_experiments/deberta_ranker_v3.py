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

### define dataloader ### 

class TripletData(Dataset):
    def __init__(self, path):
        super(TripletData, self).__init__()
        self.data = [txt for txt in Path(path).glob('*.txt')]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class custom_collate_use_metric_learning(object):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("tanapatentlm/patentdeberta_base_spec_1024_pwi") 
        self.chunk_size = 300 
    
    def clean_text(self, t):
        x = re.sub("\d+","",t)
        x = x.replace("\n"," ") 
        x = x.strip() 
        return x 
    
    def __call__(self, batch):
        input_ids, attn_masks = [], [] 
        labels = [] 
        ids = 0  
        for idx, txt_file in enumerate(batch):
            with txt_file.open("r", encoding="utf8") as f:
                data = f.read()
            triplet = data.split("\n\n\n") 
            q, p, n = triplet            
            q_splitted = q.split('\n\n') 
            p_splitted = p.split('\n\n')
            n_splitted = n.split('\n\n') 
            first_q = self.clean_text(q_splitted[0]) 
            first_p = self.clean_text(p_splitted[0]) 
            first_n = self.clean_text(n_splitted[0]) 
            
            encoded_q = encoded_inputs = self.tokenizer(first_q, return_tensors='pt', max_length=self.chunk_size, padding='max_length', truncation=True) 
            encoded_p = encoded_inputs = self.tokenizer(first_p, return_tensors='pt', max_length=self.chunk_size, padding='max_length', truncation=True)
            encoded_n = encoded_inputs = self.tokenizer(first_n, return_tensors='pt', max_length=self.chunk_size, padding='max_length', truncation=True)
            
            input_ids.append(encoded_q['input_ids']) 
            attn_masks.append(encoded_q['attention_mask']) 
            labels.append(ids*2) 
            
            input_ids.append(encoded_p['input_ids']) 
            attn_masks.append(encoded_p['attention_mask'])
            labels.append(ids*2) 
            
            input_ids.append(encoded_n['input_ids']) 
            attn_masks.append(encoded_n['attention_mask']) 
            labels.append(ids*2+1) 
        
        input_ids = torch.stack(input_ids, dim=0).squeeze(dim=1)
        attn_masks = torch.stack(attn_masks, dim=0).squeeze(dim=1)    
        labels = torch.tensor(labels, dtype=int) 
        return input_ids, attn_masks, labels
            
    
class custom_collate(object):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("tanapatentlm/patentdeberta_base_spec_1024_pwi")
        self.chunk_size = 300

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
            q_splitted = q.split('\n\n') 
            p_splitted = p.split('\n\n') 
            n_splitted = n.split('\n\n')
            
            first_q = self.clean_text(q_splitted[0]) 
            first_p = self.clean_text(p_splitted[0]) 
            first_n = self.clean_text(n_splitted[0]) 
            
            encoded_q = encoded_inputs = self.tokenizer(first_q, return_tensors='pt', max_length=self.chunk_size, padding='max_length', truncation=True) 
            
            encoded_p = encoded_inputs = self.tokenizer(first_p, return_tensors='pt', max_length=self.chunk_size, padding='max_length', truncation=True)
            
            encoded_n = encoded_inputs = self.tokenizer(first_n, return_tensors='pt', max_length=self.chunk_size, padding='max_length', truncation=True)
            
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
collate_metric_learning = custom_collate_use_metric_learning() 
train_dataloader = DataLoader(train_set, batch_size=35, num_workers=4, collate_fn=collate_metric_learning, shuffle=True)
validation_dataloader = DataLoader(valid_set, batch_size=35, num_workers=4, collate_fn=collate, shuffle=False) 

class DeBERTa_Ranker(torch.nn.Module):
    def __init__(self, continual_learning=False, dropout_prob=0.1): 
        super(DeBERTa_Ranker, self).__init__() 
        self.d_model = 768 # size of hidden dimension
        self.device = torch.device('cuda') 
        self.tokenizer = AutoTokenizer.from_pretrained("tanapatentlm/patentdeberta_base_spec_1024_pwi")
        self.model = AutoModel.from_pretrained("tanapatentlm/patentdeberta_base_spec_1024_pwi")
        self.gru = nn.GRU(self.d_model, self.d_model, batch_first=True) 
        self.dropout = nn.Dropout(dropout_prob) 
        
        if continual_learning == True:
            print("loading from previous DeBERTa Base FirstP checkpoint...")
            ckpt_path = "../storage/checkpoints/deberta_base_spec_1024_pwi/checkpoints-epoch=08-val_loss=0.06.ckpt" 
            checkpoint = torch.load(ckpt_path)
            new_weights = self.model.state_dict() 
            old_weights = list(checkpoint['state_dict'].items()) 
            i=0
            for k, _ in new_weights.items():
                new_weights[k] = old_weights[i][1]
                i += 1
            self.model.load_state_dict(new_weights) 
        
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float() 
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
    def forward(self, input_ids, attention_mask):
        x = self.model(input_ids, attention_mask)
        output, hn = self.gru(x[0])
        hn = hn.squeeze(0) 
        hn = self.dropout(hn) 
        return hn  
    
### train model ### 
model = DeBERTa_Ranker(continual_learning=True)
ckpt_name = "../storage/miner_deberta_first_claims_epoch_1_train_loss_0.04917613631956766_val_loss_0.10339764724347457.pt" 
checkpoint = torch.load(ckpt_name)
print("loading from previously trainined deberta first independent claim checkpoint:{}".format(ckpt_name)) 
model.load_state_dict(checkpoint, strict=False)
model.cuda()

accumulation_steps = 4

loss_func = losses.TripletMarginLoss() 
miner = miners.TripletMarginMiner() 

loss_func_torch = nn.TripletMarginLoss() 
optimizer = AdamW(model.parameters(), lr=1e-4) 
epochs = 3
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
            if step%1000==0 and step > 0:
                print("saving intermediate checkpoint") 
                torch.save(model.state_dict(), "../storage/DEBERTA_Ranker_v2_intermediate_checkpoint_epoch_{}_steps_{}.pt".format(epoch_i+1, step)) 
            batch = tuple(t.to(device) for t in batch) 
            input_ids, attn_masks, labels = batch
            embeddings = model(input_ids, attn_masks) 
            hard_pairs = miner(embeddings, labels) 
            loss = loss_func(embeddings, labels, hard_pairs) 
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
        loss = loss_func_torch(q_emb, p_emb, n_emb) 
        val_loss += loss.item() 
    avg_val_loss = val_loss / len(validation_dataloader) 
    print("average validation loss: {}".format(avg_val_loss)) 
    val_losses.append(avg_val_loss) 
    
    print("saving checkpoint")  
    torch.save(model.state_dict(), "../storage/DEBERTA_Ranker_v2_epoch_{}_train_loss_{}_val_loss_{}.pt".format(epoch_i+1, avg_train_loss, avg_val_loss))  
