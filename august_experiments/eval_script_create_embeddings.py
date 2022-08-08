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
import pickle 

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
        self.chunk_size = 256
    
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

### example model declaration ### 
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
        x = self.mean_pooling(x, attention_mask)
        return x # mean-pooling 
    
### train model ### 
ckpt_name = "../storage/first_claims_deberta_intermediate_checkpoint_epoch_1_steps_3000.pt" 
checkpoint = torch.load(ckpt_name) 
model = DeBERTa_Ranker(continual_learning=True)
print("loading state dict from {}...".format(ckpt_name)) 
model.load_state_dict(checkpoint) 
model.cuda()
model.eval() 

test_set = TripletData("../storage/patent_experiments/FGH_claim_triplet_v0.1s/test")
collate = custom_collate()
test_dataloader = DataLoader(test_set, batch_size=53, num_workers=4, collate_fn=collate, shuffle=True)
device = torch.device('cuda') 
loss_func = nn.TripletMarginLoss() 

q_v, p_v, n_v = [], [], [] 

test_loss = 0 
cnt = 0 
for step, batch in tqdm(enumerate(test_dataloader), desc="Testing", position=0, leave=True, total=len(test_dataloader)):
        if step == 100:
            break 
        batch = tuple(t.to(device) for t in batch) 
        q_input_ids, q_input_mask, p_input_ids, p_input_mask, n_input_ids, n_input_mask = batch 
        with torch.no_grad():
            q_emb = model(q_input_ids, q_input_mask) 
            p_emb = model(p_input_ids, p_input_mask) 
            n_emb = model(n_input_ids, n_input_mask)
            
        q_v.append(q_emb.detach().cpu().numpy().copy()) 
        p_v.append(p_emb.detach().cpu().numpy().copy()) 
        n_v.append(n_emb.detach().cpu().numpy().copy()) 
        
        loss = loss_func(q_emb, p_emb, n_emb) 
        test_loss += loss.item() 
        cnt += 1 
    
# avg_test_loss = test_loss / len(test_dataloader) 
avg_test_loss = test_loss / cnt 
print("average test loss: {}".format(avg_test_loss)) 

q_v = np.concatenate(q_v, axis=0) 
p_v = np.concatenate(p_v, axis=0) 
n_v = np.concatenate(n_v, axis=0) 
candidate = np.concatenate([p_v, n_v], axis=0) 
embeddings = {
    'query': q_v, 
    'candidate': candidate
} 

with open("../storage/3000_normal.pkl", "wb") as f:
    pickle.dump(embeddings, f) 
    
print("done!!!!!") 
