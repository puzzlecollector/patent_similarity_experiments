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
            
            qb_input_ids[idx] = encoded_q['input_ids'] 
            qb_attn_masks[idx] = encoded_q['attention_mask'] 
            pb_input_ids[idx] = encoded_p['input_ids'] 
            pb_attn_masks[idx] = encoded_p['attention_mask'] 
            nb_input_ids[idx] = encoded_n['input_ids'] 
            nb_attn_masks[idx] = encoded_n['attention_mask'] 
        return qb_input_ids, qb_attn_masks, pb_input_ids, pb_attn_masks, nb_input_ids, nb_attn_masks
    
    
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
    
### train model ### 
ckpt_name = "../storage/ipc_title_firstclaims_epoch_2_steps_6000_val_loss_0.13378801833644857.pt" 
checkpoint = torch.load(ckpt_name) 
model = DeBERTa_Ranker(continual_learning=True)
print("loading state dict from {}…".format(ckpt_name)) 
model.load_state_dict(checkpoint) 
model.cuda()
model.eval() 

test_set = TripletData("../storage/test_spec")
collate = custom_collate()
test_dataloader = DataLoader(test_set, batch_size=80, num_workers=4, collate_fn=collate, shuffle=False)
device = torch.device('cuda') 
loss_func = nn.TripletMarginLoss() 

q_v, p_v, n_v = [], [], [] 

test_loss = 0 
cnt = 0 
for step, batch in tqdm(enumerate(test_dataloader), desc="Testing", position=0, leave=True, total=len(test_dataloader)):
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

with open("../storage/ipc_title_firstclaim.pkl", "wb") as f:
    pickle.dump(embeddings, f) 
    
print("done!!!!!")
