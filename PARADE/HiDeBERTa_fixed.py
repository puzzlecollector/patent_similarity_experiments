import numpy as np 
import pandas as pd
import torch 
import torch.nn.functional as F
import torch.nn as nn 
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup 
from tqdm import tqdm 
from torch import Tensor 
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler 
import math 
import time 
import datetime 
import os 
import re 
from sklearn.model_selection import train_test_split, StratifiedKFold
import logging
import addict

### ignore warning
def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)

            
set_global_logging_level(logging.ERROR, ["transformers", "nlp", "torch", "tensorflow", "tensorboard", "wandb"])

tokenizer = AutoTokenizer.from_pretrained("tanapatentlm/patentdeberta_base_spec_1024_pwi") 

train_files = os.listdir("../storage/patent_experiments/FGH_claim_triplet_v0.1s/train") 
valid_files = os.listdir("../storage/patent_experiments/FGH_claim_triplet_v0.1s/valid") 

print(len(train_files), len(valid_files)) 

train_queries, train_positives, train_negatives = [], [], [] 

for i in tqdm(range(len(train_files[:2000])), position=0, leave=True):
    with open("../storage/patent_experiments/FGH_claim_triplet_v0.1s/train/" + train_files[i], "r") as f:
        data = f.read() 
    triplet = data.split("\n\n\n") 
    q, p, n = triplet 
    train_queries.append(q) 
    train_positives.append(p)   
    train_negatives.append(n) 
    
    
valid_queries, valid_positives, valid_negatives = [], [], []

for i in tqdm(range(len(valid_files[:2000])), position=0, leave=True):
    with open("../storage/patent_experiments/FGH_claim_triplet_v0.1s/valid/" + valid_files[i], "r") as f:
        data = f.read() 
    triplet = data.split("\n\n\n") 
    q, p, n = triplet 
    valid_queries.append(q) 
    valid_positives.append(p) 
    valid_negatives.append(n) 
    
def chunk_tokens(tokens, start_token_id, end_token_id, overlap=12, chunk_size=128):
    chunk_size = chunk_size - tokenizer.num_special_tokens_to_add() 
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

def get_chunked(t, seq_len = 10, window_size = 128): 
    encoded_inputs = tokenizer(t, add_special_tokens=False) 
    input_ids = encoded_inputs['input_ids'] 
    attention_masks = encoded_inputs['attention_mask'] 
    chunks = chunk_tokens(input_ids, tokenizer.cls_token_id, tokenizer.sep_token_id) 
    chunk_attention_masks = chunk_tokens(attention_masks, 1, 1) 
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

train_q_input_ids, train_q_attn_masks = [], [] 
train_p_input_ids, train_p_attn_masks = [], [] 
train_n_input_ids, train_n_attn_masks = [], [] 

for i in tqdm(range(len(train_files[:2000]))):
    q_input_ids, q_attn_masks = get_chunked(train_queries[i]) 
    train_q_input_ids.append(q_input_ids) 
    train_q_attn_masks.append(q_attn_masks)  
    
    p_input_ids, p_attn_masks = get_chunked(train_positives[i]) 
    train_p_input_ids.append(p_input_ids) 
    train_p_attn_masks.append(p_attn_masks) 
    
    n_input_ids, n_attn_masks = get_chunked(train_negatives[i]) 
    train_n_input_ids.append(n_input_ids) 
    train_n_attn_masks.append(n_attn_masks) 

train_q_input_ids = torch.stack(train_q_input_ids, dim=0) 
train_q_attn_masks = torch.stack(train_q_attn_masks, dim=0) 
train_p_input_ids = torch.stack(train_p_input_ids, dim=0) 
train_p_attn_masks = torch.stack(train_p_attn_masks, dim=0) 
train_n_input_ids = torch.stack(train_n_input_ids, dim=0) 
train_n_attn_masks = torch.stack(train_n_attn_masks, dim=0) 

print(train_q_input_ids.shape, train_q_attn_masks.shape, train_p_input_ids.shape, train_p_attn_masks.shape, train_n_input_ids.shape, train_n_attn_masks.shape) 

valid_q_input_ids, valid_q_attn_masks = [], [] 
valid_p_input_ids, valid_p_attn_masks = [], [] 
valid_n_input_ids, valid_n_attn_masks = [], [] 

for i in tqdm(range(len(valid_files[:2000]))):
    q_input_ids, q_attn_masks = get_chunked(valid_queries[i]) 
    valid_q_input_ids.append(q_input_ids) 
    valid_q_attn_masks.append(q_attn_masks)  
    
    p_input_ids, p_attn_masks = get_chunked(valid_positives[i]) 
    valid_p_input_ids.append(p_input_ids) 
    valid_p_attn_masks.append(p_attn_masks) 
    
    n_input_ids, n_attn_masks = get_chunked(valid_negatives[i]) 
    valid_n_input_ids.append(n_input_ids) 
    valid_n_attn_masks.append(n_attn_masks) 
    
valid_q_input_ids = torch.stack(valid_q_input_ids, dim=0) 
valid_q_attn_masks = torch.stack(valid_q_attn_masks, dim=0) 
valid_p_input_ids = torch.stack(valid_p_input_ids, dim=0) 
valid_p_attn_masks = torch.stack(valid_p_attn_masks, dim=0) 
valid_n_input_ids = torch.stack(valid_n_input_ids, dim=0) 
valid_n_attn_masks = torch.stack(valid_n_attn_masks, dim=0) 


print(valid_q_input_ids.shape, valid_q_attn_masks.shape, valid_p_input_ids.shape, valid_p_attn_masks.shape, valid_n_input_ids.shape, valid_n_attn_masks.shape) 


## create dataloader 
batch_size = 2
train_data = TensorDataset(train_q_input_ids, train_q_attn_masks, 
                           train_p_input_ids, train_p_attn_masks, 
                           train_n_input_ids, train_n_attn_masks) 
train_sampler = RandomSampler(train_data) 
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size) 

validation_data = TensorDataset(valid_q_input_ids, valid_q_attn_masks,
                                valid_p_input_ids, valid_p_attn_masks, 
                                valid_n_input_ids, valid_n_attn_masks) 
validation_sampler = SequentialSampler(validation_data) 
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size) 


class AttentivePooling(torch.nn.Module): 
    def __init__(self, input_dim): 
        super(AttentivePooling, self).__init__() 
        self.W = nn.Linear(input_dim, 1) 
    def forward(self, x):
        ''' input: [batch_size, seq_len, hidden_dim] '''
        softmax = F.softmax 
        att_w = softmax(self.W(x).squeeze(-1)).unsqueeze(-1) 
        x = torch.sum(x * att_w, dim=1) 
        return x 
    
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
        self.chunk_LM = AutoModel.from_pretrained("tanapatentlm/patentdeberta_base_spec_1024_pwi")
        
        print("loading from checkpoint...")
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
        encoder_layers = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=8, batch_first=True) 
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=6)
        self.attentive_pooling = AttentivePooling(input_dim=self.d_model) 
        
    def forward(self, input_ids, attention_mask):
        x = self.chunk_LM(input_ids=input_ids[:,0,:], attention_mask=attention_mask[:,0,:]) 
        x = x[0][:,0,:]
        x = torch.unsqueeze(x, dim=1) 
        for i in range(1, self.seq_len):
            output_vec = self.chunk_LM(input_ids=input_ids[:,i,:], attention_mask=attention_mask[:,i,:]) 
            output_vec = output_vec[0][:,0,:] 
            output_vec = torch.unsqueeze(output_vec, dim=1) 
            x = torch.cat((x, output_vec), dim=1)  
        x = self.pos_encoder(x.to(self.device)) 
        x = self.transformer_encoder(x) 
        x = self.attentive_pooling(x) 
        return x 
    
### train model ### 
model = DeBERTa_Ranker()
model.cuda()

loss_func = nn.TripletMarginWithDistanceLoss(distance_function = lambda x, y: 1.0 - F.cosine_similarity(x,y))  
optimizer = AdamW(model.parameters(), lr=1e-4) 
epochs = 5
total_steps = len(train_dataloader) * epochs 
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=0, 
                                            num_training_steps=total_steps) 

train_losses, val_losses = [], []
device = torch.device('cuda') 
model.zero_grad() 

for epoch_i in tqdm(range(epochs), desc="Epochs", position=0, leave=True, total=epochs):
    train_loss = 0.0 
    model.train()
    with tqdm(train_dataloader, unit="batch") as tepoch:
        for step, batch in enumerate(tepoch):
            batch = tuple(t.to(device) for t in batch) 
            q_input_ids, q_input_mask, p_input_ids, p_input_mask, n_input_ids, n_input_mask = batch 
            q_emb = model(q_input_ids, q_input_mask) 
            p_emb = model(p_input_ids, p_input_mask) 
            n_emb = model(n_input_ids, n_input_mask) 
            loss = loss_func(q_emb, p_emb, n_emb) 
            train_loss += loss.item() 
            loss.backward() 
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
            optimizer.step() 
            scheduler.step() 
            model.zero_grad() 
            tepoch.set_postfix(loss=train_loss/(step+1)) 
            time.sleep(0.1) 
    avg_train_loss = train_loss / len(train_dataloader) 
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
    torch.save(model.state_dict(), "../storage/DeBERTa_Ranker_epoch:{}_val_loss:{}.pt".format(epoch_i+1, avg_val_loss)) 
