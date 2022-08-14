import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup, BigBirdPegasusForConditionalGeneration
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
        self.pegasus_tokenizer = AutoTokenizer.from_pretrained("google/bigbird-pegasus-large-bigpatent")
        self.tokenizer = AutoTokenizer.from_pretrained("tanapatentlm/patentdeberta_base_spec_1024_pwi")
        self.pegasus = BigBirdPegasusForConditionalGeneration.from_pretrained("google/bigbird-pegasus-large-bigpatent") 
        self.pegasus.cuda() 
        self.pegasus.eval() 
        self.chunk_size = 512
        self.device = torch.device('cuda') 
        
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
        for batch_idx, txt_file in enumerate(batch):
            with txt_file.open("r", encoding="utf8") as f:
                data = f.read()
            triplet = data.split("\n\n\n") 
            q, p, n = triplet 
            q_ttl = re.search("<TTL>([\s\S]*?)<IPC>", q).group(1) 
            q_ipc = re.search("<IPC>([\s\S]*?)<ABST>", q).group(1) 
            q_clms = re.search("<CLMS>([\s\S]*?)<DESC>", q).group(1) 
            q_desc = re.search("<DESC>([\s\S]*)$", q).group(1)
            ### process query text ### 
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
            # summarize description 
            splitted_q_desc = q_desc.split("\n")
            detailed_desc_idx = -1 
            found = False 
            for idx, q_txt in enumerate(splitted_q_desc):
                if q_txt.isupper(): 
                    if "DETAILED DESCRIPTION" in q_txt:
                        detailed_desc_idx = idx 
                        found = True 
                if found == True:
                    break 
            if found == False:
                q_summ_desc = q_desc 
            else:
                q_summ_desc = ' '.join(splitted_q_desc[detailed_desc_idx+1:]) 
            encoded_q_summ_desc = self.pegasus_tokenizer(q_summ_desc, return_tensors='pt', max_length=4096, padding='max_length', truncation=True).to(self.device) 
            with torch.no_grad(): 
                prediction = self.pegasus.generate(**encoded_q_summ_desc)
                prediction = self.pegasus_tokenizer.batch_decode(prediction)[0]
                prediction = prediction.replace("</s>","").replace("<s>","")
            q_text_input = q_ipc + " " + q_ttl + self.tokenizer.sep_token + selected_q_clm + self.tokenizer.sep_token + prediction
            encoded_q = self.tokenizer(q_text_input, return_tensors='pt', max_length=self.chunk_size, padding='max_length', truncation=True)
            
            
            
            ### process positive text ### 
            p_ttl = re.search("<TTL>([\s\S]*?)<IPC>", p).group(1) 
            p_ipc = re.search("<IPC>([\s\S]*?)<ABST>", p).group(1) 
            p_clms = re.search("<CLMS>([\s\S]*?)<DESC>", p).group(1)
            p_desc = re.search("<DESC>([\s\S]*)$", p).group(1)
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
            # summarize description 
            splitted_p_desc = p_desc.split("\n") 
            detailed_desc_idx = -1 
            found = False
            for idx, p_txt in enumerate(splitted_p_desc):
                if p_txt.isupper(): 
                    if "DETAILED DESCRIPTION" in p_txt:
                        detailed_desc_idx = idx 
                        found = True 
                if found == True:
                    break 
            if found == False:
                p_summ_desc = p_desc 
            else:
                p_summ_desc = ' '.join(splitted_p_desc[detailed_desc_idx+1:])
            encoded_p_summ_desc = self.pegasus_tokenizer(p_summ_desc, return_tensors='pt', max_length=4096, padding='max_length', truncation=True).to(self.device) 
            with torch.no_grad(): 
                prediction = self.pegasus.generate(**encoded_p_summ_desc)
                prediction = self.pegasus_tokenizer.batch_decode(prediction)[0]
                prediction = prediction.replace("</s>","").replace("<s>","") 
            p_text_input = p_ipc + " " + p_ttl + self.tokenizer.sep_token + selected_p_clm + self.tokenizer.sep_token + prediction
            encoded_p = self.tokenizer(p_text_input, return_tensors='pt', max_length=self.chunk_size, padding='max_length', truncation=True) 
            
            ### process negative text ### 
            n_ttl = re.search("<TTL>([\s\S]*?)<IPC>", n).group(1) 
            n_ipc = re.search("<IPC>([\s\S]*?)<ABST>", n).group(1)
            n_clms = re.search("<CLMS>([\s\S]*?)<DESC>", n).group(1)
            n_desc = re.search("<DESC>([\s\S]*)$", n).group(1)
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
            
            splitted_n_desc = n_desc.split("\n") 
            detailed_desc_idx = -1 
            found = False
            for idx, n_txt in enumerate(splitted_n_desc):
                if n_txt.isupper(): 
                    if "DETAILED DESCRIPTION" in n_txt:
                        detailed_desc_idx = idx 
                        found = True 
                if found == True:
                    break 
            if found == False:
                n_summ_desc = n_desc 
            else:
                n_summ_desc = ' '.join(splitted_n_desc[detailed_desc_idx+1:])
            encoded_n_summ_desc = self.pegasus_tokenizer(n_summ_desc, return_tensors='pt', max_length=4096, padding='max_length', truncation=True).to(self.device) 
            with torch.no_grad(): 
                prediction = self.pegasus.generate(**encoded_n_summ_desc)
                prediction = self.pegasus_tokenizer.batch_decode(prediction)[0]
                prediction = prediction.replace("</s>","").replace("<s>","")  
            n_text_input = n_ipc + " " + n_ttl + self.tokenizer.sep_token + selected_n_clm + self.tokenizer.sep_token + prediction
            encoded_n = self.tokenizer(n_text_input, return_tensors='pt', max_length=self.chunk_size, padding='max_length', truncation=True) 
            
            qb_input_ids[batch_idx] = encoded_q['input_ids'] 
            qb_attn_masks[batch_idx] = encoded_q['attention_mask'] 
            pb_input_ids[batch_idx] = encoded_p['input_ids'] 
            pb_attn_masks[batch_idx] = encoded_p['attention_mask'] 
            nb_input_ids[batch_idx] = encoded_n['input_ids'] 
            nb_attn_masks[batch_idx] = encoded_n['attention_mask'] 
        return qb_input_ids, qb_attn_masks, pb_input_ids, pb_attn_masks, nb_input_ids, nb_attn_masks

train_set = TripletData("../storage/train_spec")
collate = custom_collate()
train_dataloader = DataLoader(train_set, batch_size=20, collate_fn=collate, shuffle=True)

tokenizer = AutoTokenizer.from_pretrained("tanapatentlm/patentdeberta_base_spec_1024_pwi")

for step, batch in enumerate(tqdm(train_dataloader)):
    q_input_ids, q_attn_masks, p_input_ids, p_attn_masks, n_input_ids, n_attn_masks = batch 
    print(q_input_ids.shape) 
    
    print(tokenizer.decode(q_input_ids[5]))
    
    print("================") 
    print(tokenizer.decode(p_input_ids[5]))
    
    print("================")
    print(tokenizer.decode(n_input_ids[5]))
    
    break 