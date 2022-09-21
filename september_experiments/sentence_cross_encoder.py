import numpy as np 
import pandas as pd 
import os 
from tqdm.auto import tqdm 
from transformers import (
    AdamW, 
    AutoConfig, 
    AutoModel, 
    AutoTokenizer, 
    get_linear_schedule_with_warmup
) 
import torch 
import torch.nn.functional as F 
import torch.nn as nn 
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, RandomSampler, SequentialSampler, IterableDataset
import math 
import time 
import datetime 
import re

df = pd.read_excel("0919_라벨링세트_9주차_병합.xlsx")

df = df.loc[df["라벨링"].notnull(), ["쿼리 번호", "IPC 분류", "쿼리 문장", "후보 문장", "쿼리 문서 번호", "Positive 문서 번호", "라벨링"]] 
df = df.dropna() 
labels_fixed = [] 
labels = df["라벨링"].values 

for i in range(len(labels)): 
    if labels[i] == 0.1:
        labels_fixed.append(1.0) 
    elif labels[i] not in [0,0.5,0.8,1.0]: 
        labels_fixed.append(None) 
    else: 
        labels_fixed.append(labels[i]) 
        
df["라벨링"] = labels_fixed
df = df.dropna() 
ipc_types = df["IPC 분류"].values 
unique_ipcs = np.unique(ipc_types) 

train_size = int(len(unique_ipcs) * 0.8) 
val_size = int(len(unique_ipcs) * 0.1) 

train_unique_ipcs = unique_ipcs[:train_size] 
val_unique_ipcs = unique_ipcs[train_size:train_size+val_size] 
test_unique_ipcs = unique_ipcs[train_size+val_size:] 

# make sure to only test with samples with at least one 0.8 or 1.0 score. 
train_queries, train_candidates, train_labels = [], [], [] 
valid_queries, valid_candidates, valid_labels = [], [], [] 
test_queries, test_candidates, test_labels = [], [], [] 
test_query_nums, test_candidate_nums = [], [] 

ipcs = df["IPC 분류"].values 
queries = df["쿼리 문장"].values 
candidates = df["후보 문장"].values 
labels = df["라벨링"].values
query_nums = df["쿼리 문서 번호"].values 
positive_nums = df["Positive 문서 번호"].values 

for i in tqdm(range(len(queries)), position=0, leave=True): 
    if ipcs[i] in train_unique_ipcs: 
        train_queries.append(queries[i]) 
        train_candidates.append(candidates[i]) 
        train_labels.append(labels[i]) 
    elif ipcs[i] in val_unique_ipcs: 
        valid_queries.append(queries[i]) 
        valid_candidates.append(candidates[i]) 
        valid_labels.append(labels[i]) 
    elif ipcs[i] in test_unique_ipcs: 
        test_queries.append(queries[i]) 
        test_candidates.append(candidates[i]) 
        test_labels.append(labels[i])  
        test_query_nums.append(query_nums[i]) 
        test_candidate_nums.append(positive_nums[i]) 
        
## Define Model  
## we will use a cross encoder for sentence ranking 
class SentenceRanker(nn.Module): 
    def __init__(self, plm="tanapatentlm/patentdeberta_large_spec_128_pwi"): 
        super(SentenceRanker, self).__init__() 
        self.config = AutoConfig.from_pretrained(plm)  
        self.net = AutoModel.from_pretrained(plm) 
        self.tokenizer = AutoTokenizer.from_pretrained(plm) 
        self.tokenizer.add_special_tokens({"additional_special_tokens":["[IPC]", "[TTL]", "[CLMS]", "[ABST]"]}) 
        self.net.resize_token_embeddings(len(self.tokenizer))
        self.dropout = nn.Dropout(0.1) 
        self.fc = nn.Linear(self.config.hidden_size, 1) 
        
    def mean_pooling(self, model_output, attention_mask): 
        token_embeddings = model_output[0] 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float() 
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def forward(self, input_ids, attention_mask): 
        x = self.net(input_ids, attention_mask) 
        x = self.mean_pooling(x, attention_mask) 
        x = self.dropout(x) 
        x = self.fc(x) 
        return x 
    
## Define Loss function 
class RMSELoss(nn.Module): 
    def __init__(self): 
        super().__init__()
        self.mse = nn.MSELoss() 
    def forward(self, yhat, y): 
        return torch.sqrt(self.mse(yhat, y)) 
    
    
## Transfer learning from document ranker model 
ckpt = "epoch_end_checkpoints-epoch=00-val_loss=0.20442404.ckpt" 
checkpoint = torch.load(ckpt) 
model = SentenceRanker() 
new_weights = model.state_dict() 
old_weights = list(checkpoint["state_dict"].items())
i = 0
for j in range(len(old_weights)): 
    new_weights[old_weights[j][0]] = old_weights[j][1] 
    i += 1 
print(model.load_state_dict(new_weights)) 

tokenizer = AutoTokenizer.from_pretrained("tanapatentlm/patentdeberta_large_spec_128_pwi")

train_input_ids, train_attn_masks = [], [] 
valid_input_ids, valid_attn_masks = [], [] 

for i in tqdm(range(len(train_queries)), position=0, leave=True): 
    encoded_input = tokenizer(train_queries[i], train_candidates[i], max_length=512, truncation=True, padding="max_length") 
    train_input_ids.append(encoded_input["input_ids"]) 
    train_attn_masks.append(encoded_input["attention_mask"]) 

for i in tqdm(range(len(valid_queries)), position=0, leave=True): 
    encoded_input = tokenizer(valid_queries[i], valid_candidates[i], max_length=512, truncation=True, padding="max_length") 
    valid_input_ids.append(encoded_input["input_ids"])
    valid_attn_masks.append(encoded_input["attention_mask"]) 
    
train_input_ids = torch.tensor(train_input_ids, dtype=int) 
train_attn_masks = torch.tensor(train_attn_masks, dtype=int) 
train_labels = torch.tensor(train_labels).float() 

valid_input_ids = torch.tensor(valid_input_ids, dtype=int) 
valid_attn_masks = torch.tensor(valid_attn_masks, dtype=int) 
valid_labels = torch.tensor(valid_labels).float() 

print(train_input_ids.shape, train_attn_masks.shape, train_labels.shape, valid_input_ids.shape, valid_attn_masks.shape, valid_labels.shape)

batch_size = 20

train_data = TensorDataset(train_input_ids, train_attn_masks, train_labels) 
train_sampler = RandomSampler(train_data) 
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size) 

val_data = TensorDataset(valid_input_ids, valid_attn_masks, valid_labels) 
val_sampler = SequentialSampler(val_data) 
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size) 

loss_func = RMSELoss() 
model.cuda() 
optimizer = AdamW(model.parameters(), lr=2e-5) 
epochs = 10 
total_steps = len(train_dataloader) * epochs 
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=100,
                                            num_training_steps=total_steps) 
device = torch.device("cuda") 
model.zero_grad() 
for epcoh_i in tqdm(range(0, epochs), desc="Epochs", position=0, leave=True, total=epochs): 
    train_loss = 0 
    model.train() 
    with tqdm(train_dataloader, unit="batch") as tepoch: 
        for step, batch in enumerate(tepoch): 
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_masks, b_labels = batch 
            outputs = model(b_input_ids, b_input_masks) 
            loss = loss_func(outputs, b_labels) 
            train_loss += loss.item() 
            loss.backward() 
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
            optimizer.step() 
            scheduler.step() 
            model.zero_grad() 
            tepoch.set_postfix(loss=train_loss / (step+1)) 
            time.sleep(0.1) 
    avg_train_loss = train_loss / len(train_dataloader) 
    print(f"average train loss : {avg_train_loss}")
    
    val_loss = 0 
    model.eval() 
    for step, batch in tqdm(enumerate(val_dataloader), desc="Validating", position=0, leave=True, total=len(val_dataloader)): 
        batch = tuple(t.to(device) for t in batch) 
        b_input_ids, b_input_masks, b_labels = batch 
        with torch.no_grad(): 
            outputs = model(b_input_+ids, b_input_masks) 
        loss = loss_func(outputs, b_labels) 
        val_loss += loss.item() 
    avg_val_loss = val_loss / len(val_dataloader) 
    print(f"average validation loss : {avg_val_loss}") 
    val_losses.append(avg_val_loss) 
    
    if np.min(val_losses) == val_losses[-1]:
        torch.save(model.state_dict(), "DeBERTa_Cross_Encoder.pt")
