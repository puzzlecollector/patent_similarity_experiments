# with 1 for positive pairs, 0.75 for hard-negative pairs and 0.0 for negative pairs 
import numpy as np 
import pandas as pd
import torch 
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
import pickle 

class TripletData(Dataset):
    def __init__(self, path):
        super(TripletData, self).__init__() 
        self.data = [txt for txt in Path(path).glob("*.txt")] 
    def __getitem__(self, index):
        return self.data[index] 
    def __len__(self):
        return len(self.data) 

class custom_collate(object):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("tanapatentlm/patentdeberta_base_spec_1024_pwi") 
        self.chunk_size = 256 
    def clean_text(self, t):
        x = re.sub("\d+.","", t) 
        x = x.replace("\n"," ") 
        x = x.strip() 
        return x 
    def __call__(self, batch):
        b = len(batch) 
        x1_input_ids, x2_input_ids = [],[]
        x1_attn_masks, x2_attn_masks = [],[]
        labels = [] # 1.0 for positive, 0.75 for hard-negative, 0.0 for negative 
        
        triplet_input_ids = [] 
        triplet_attn_masks = [] 
        
        for idx, txt_file in enumerate(batch):
            with txt_file.open("r", encoding="utf8") as f:
                data = f.read() 
            triplet = data.split("\n\n\n") 
            q, p, n = triplet 
            q_ttl = re.search("<TTL>([\s\S]*?)<IPC>", q).group(1) 
            q_ipc = re.search("<IPC>([\s\S]*?)<ABST>", q).group(1) 
            q_clms = re.search("<CLMS>([\s\S]*?)<DESC>", q).group(1) 
            q_ttl = q_ttl.lower() # convert title to lower case 
            q_ipc = q_ipc[:3] # get first three characters 
            # get first claim as long as it is not canceled 
            q_ind_clms = q_clms.split('\n\n') 
            selected_q_clm = q_ind_clms[0] 
            for q_ind_clm in q_ind_clms:
                if '(canceled)' in q_ind_clm:
                    continue
                else:
                    selected_q_clm = q_ind_clm
                    break 
            selected_q_clm = self.clean_text(selected_q_clm)
            q_text_input = q_ipc + " " + q_ttl + self.tokenizer.sep_token + selected_q_clm
            encoded_q = self.tokenizer(q_text_input, return_tensors='pt', max_length=self.chunk_size, padding='max_length', truncation=True)
            
            p_ttl = re.search("<TTL>([\s\S]*?)<IPC>", p).group(1) 
            p_ipc = re.search("<IPC>([\s\S]*?)<ABST>", p).group(1) 
            p_clms = re.search("<CLMS>([\s\S]*?)<DESC>", p).group(1)
            p_ttl = p_ttl.lower() 
            p_ipc = p_ipc[:3] 
            p_ind_clms = p_clms.split("\n\n") 
            selected_p_clm = p_ind_clms[0] 
            for p_ind_clm in p_ind_clms:
                if '(canceled)' in p_ind_clm:
                    continue
                else:
                    selected_p_clm = p_ind_clm
                    break 
            selected_p_clm = self.clean_text(selected_p_clm) 
            p_text_input = p_ipc + " " + p_ttl + self.tokenizer.sep_token + selected_p_clm 
            encoded_p = self.tokenizer(p_text_input, return_tensors='pt', max_length=self.chunk_size, padding='max_length', truncation=True) 
            
            n_ttl = re.search("<TTL>([\s\S]*?)<IPC>", n).group(1) 
            n_ipc = re.search("<IPC>([\s\S]*?)<ABST>", n).group(1)
            n_clms = re.search("<CLMS>([\s\S]*?)<DESC>", n).group(1)
            n_ttl = n_ttl.lower() 
            n_ipc = n_ipc[:3] 
            n_ind_clms = n_clms.split("\n\n") 
            selected_n_clm = n_ind_clms[0] 
            for n_ind_clm in n_ind_clms:
                if '(canceled)' in n_ind_clm:
                    continue 
                else:
                    selected_n_clm = n_ind_clm
                    break 
            selected_n_clm = self.clean_text(selected_n_clm) 
            n_text_input = n_ipc + " " + n_ttl + self.tokenizer.sep_token + selected_n_clm 
            encoded_n = self.tokenizer(n_text_input, return_tensors='pt', max_length=self.chunk_size, padding='max_length', truncation=True) 
            triplet_input_ids.append((encoded_q['input_ids'],
                                      encoded_p['input_ids'],
                                      encoded_n['input_ids']))
            triplet_attn_masks.append((encoded_q['attention_mask'], 
                                       encoded_p['attention_mask'],
                                       encoded_n['attention_mask'])) 
        # create pairs with scores 
        checked_pairs = set() 
        for i in range(len(triplet_input_ids)):
            for j in range(len(triplet_input_ids)):
                if (i,j) in checked_pairs:
                    continue 
                checked_pairs.add((i,j))
                checked_pairs.add((j,i)) 
                if i == j: 
                    x1_input_ids.append(triplet_input_ids[i][0])
                    x1_attn_masks.append(triplet_attn_masks[i][0])
                    x2_input_ids.append(triplet_input_ids[i][1]) 
                    x2_attn_masks.append(triplet_attn_masks[i][1]) 
                    labels.append(1.0)
                    
                    x1_input_ids.append(triplet_input_ids[i][0]) 
                    x1_attn_masks.append(triplet_attn_masks[i][0])
                    x2_input_ids.append(triplet_input_ids[i][2]) 
                    x2_attn_masks.append(triplet_attn_masks[i][2]) 
                    labels.append(0.75) 
                elif i != j:
                    for k in range(3):
                        for l in range(3):
                            x1_input_ids.append(triplet_input_ids[i][k]) 
                            x1_attn_masks.append(triplet_input_ids[i][k]) 
                            x2_input_ids.append(triplet_input_ids[j][l]) 
                            x2_attn_masks.append(triplet_input_ids[j][l]) 
                            labels.append(0.0) 
        x1_input_ids = torch.stack(x1_input_ids, dim=0).squeeze(dim=1)  
        x1_attn_masks = torch.stack(x1_attn_masks, dim=0).squeeze(dim=1) 
        x2_input_ids = torch.stack(x2_input_ids, dim=0).squeeze(dim=1) 
        x2_attn_masks = torch.stack(x2_attn_masks, dim=0).squeeze(dim=1) 
        labels = torch.tensor(labels).float()  
        return x1_input_ids, x1_attn_masks, x2_input_ids, x2_attn_masks, labels 
    
tokenizer = AutoTokenizer.from_pretrained("tanapatentlm/patentdeberta_base_spec_1024_pwi") 
train_set = TripletData("../storage/train_spec")
valid_set = TripletData("../storage/valid_spec") 
collate = custom_collate()
train_dataloader = DataLoader(train_set, batch_size=4, collate_fn=collate, shuffle=True) 
validation_dataloader = DataLoader(valid_set, batch_size=4, collate_fn=collate, shuffle=False) 

### define model ### 
class DeBERTa_Ranker(torch.nn.Module):
    def __init__(self, continual_learning=False): 
        super(DeBERTa_Ranker, self).__init__() 
        self.d_model = 768 
        self.device = torch.device('cuda') 
        self.tokenizer = AutoTokenizer.from_pretrained("tanapatentlm/patentdeberta_base_spec_1024_pwi")
        self.model = AutoModel.from_pretrained("tanapatentlm/patentdeberta_base_spec_1024_pwi")
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
        x = self.mean_pooling(x, attention_mask) # mean-pooling 
        return x 

### define loss function ### 
class CosineSimilarityLoss(nn.Module):
    def __init__(self, loss_fct = nn.MSELoss(), cos_score_transformation=nn.Identity()):
        super(CosineSimilarityLoss, self).__init__()
        self.loss_fct = loss_fct
        self.cos_score_transformation = cos_score_transformation
    def forward(self, x1_embs, x2_embs, labels):
        output = self.cos_score_transformation(torch.cosine_similarity(x1_embs, x2_embs))
        return self.loss_fct(output, labels.view(-1))

### load from previous MRR@100 0.25 checkpoint ### 
checkpoint = torch.load("continued_miner_deberta_first_claims_epoch_1_steps_1500_val_loss_0.10022615737521476.pt")
model = DeBERTa_Ranker(continual_learning=True)
print("loading from previous checkpoint...") 
model.load_state_dict(checkpoint) 
model.cuda() 

accumulation_steps = 4
optimizer = AdamW(model.parameters(), lr=1e-4) 
epochs = 3
# total_steps = len(train_dataloader) * epochs // accumulation_steps  
total_steps = len(train_dataloader) * epochs 
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=int(0.1*total_steps), 
                                            num_training_steps=total_steps) 

loss_func = CosineSimilarityLoss() 
device = torch.device('cuda') 
model.zero_grad() 

for epoch_i in tqdm(range(epochs), desc="Epochs", position=0, leave=True, total=epochs):
    train_loss = 0.0 
    model.train()
    iters = 0 # keeps track of number of model updates based on the accumulation steps 
    with tqdm(train_dataloader, unit="batch") as tepoch:
        for step, batch in enumerate(tepoch): 
            if step%10000 == 0 and step > 0:
                val_losses = 0.0 
                model.eval() 
                for val_step, val_batch in tqdm(enumerate(validation_dataloader), desc="Validating", position=0, leave=True, total=len(validation_dataloader)):
                    val_batch = tuple(t.to(device) for t in val_batch)
                    val_b_x1_input_ids, val_b_x1_attn_masks, val_b_x2_input_ids, val_b_x2_attn_masks, val_b_labels = val_batch
                    with torch.no_grad():
                        val_x1_emb = model(val_b_x1_input_ids, val_b_x1_attn_masks) 
                        val_x2_emb = model(val_b_x2_input_ids, val_b_x2_attn_masks) 
                    val_loss = loss_func(val_x1_emb, val_x2_emb, val_b_labels)  
                    val_losses += val_loss.item() 
                avg_val_loss = val_loss / len(validation_dataloader) 
                print("saving checkpoint...") 
                torch.save(model.state_dict(), "../storage/DeBERTa_Base_CosineSim_epoch_{}_step_{}_val_loss_{}.pt".format(epoch_i+1, step, avg_val_loss))  
                model.train() # back to train mode
            
            batch = tuple(t.to(device) for t in batch) 
            b_x1_input_ids, b_x1_attn_masks, b_x2_input_ids, b_x2_attn_masks, b_labels = batch  
            x1_emb = model(b_x1_input_ids, b_x1_attn_masks) 
            x2_emb = model(b_x2_input_ids, b_x2_attn_masks) 
            loss = loss_func(x1_emb, x2_emb, b_labels) 
            loss.backward() 
            train_loss += loss.item() 
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
            optimizer.step() 
            scheduler.step() 
            model.zero_grad() 
            tepoch.set_postfix(loss=train_loss/(step+1)) 
            time.sleep(0.1) 
    avg_train_loss = train_loss / len(train_dataloader)  
    print("average train loss : {}".format(avg_train_loss)) 
    train_losses.append(avg_train_loss) 
    
    # this is the corect validation loop  
    val_loss = 0.0 
    model.eval() 
    for step, batch in tqdm(enumerate(validation_dataloader), desc="Validating", position=0, leave=True, total=len(validation_dataloader)):
        batch = tuple(t.to(device) for t in batch)
        b_x1_input_ids, b_x1_attn_masks, b_x2_input_ids, b_x2_attn_masks, b_labels = batch 
        with torch.no_grad():
            x1_emb = model(b_x1_input_ids, b_x1_attn_masks) 
            x2_emb = model(b_x2_input_ids, b_x2_attn_masks) 
        loss = loss_func(x1_emb, x2_emb, b_labels) 
        val_loss += loss.item() 
    avg_val_loss = val_loss / len(validation_dataloader) 
    print("average validation loss: {}".format(avg_val_loss)) 
    val_losses.append(avg_val_loss) 
    
    print("saving checkpoint")  
    torch.save(model.state_dict(), "../storage/DeBERTa_Base_CosineSim_epoch_{}_train_loss_{}_val_loss_{}.pt".format(epoch_i+1, avg_train_loss, avg_val_loss)) 

