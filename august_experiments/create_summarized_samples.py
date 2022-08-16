from transformers import AutoTokenizer, BigBirdPegasusForConditionalGeneration 
import torch
import os 
import pandas as pd 
import numpy as np 
import re 
from tqdm import tqdm 

files = os.listdir("../storage/train_spec") 

# define BigBirdPegasus 
tokenizer = AutoTokenizer.from_pretrained("google/bigbird-pegasus-large-bigpatent") 
pegasus = BigBirdPegasusForConditionalGeneration.from_pretrained("google/bigbird-pegasus-large-bigpatent") 
pegasus.eval() 
pegasus.cuda() 

device = torch.device("cuda") 

def clean_text(t):
    x = re.sub("\d+.","", t) 
    x = x.replace("\n"," ") 
    x = x.strip() 
    return x 


ipcs, titles, first_claims, descriptions = [],[],[],[] 

for i in tqdm(range(100), position=0, leave=True): 
    with open("../storage/train_spec/" + files[i], "r") as f:
        data = f.read() 


    q, p, n = data.split("\n\n\n") 
    
    #####################
    ### process query ###
    #####################
    q_ttl = re.search("<TTL>([\s\S]*?)<IPC>", q).group(1) 
    q_ipc = re.search("<IPC>([\s\S]*?)<ABST>", q).group(1) 
    q_clms = re.search("<CLMS>([\s\S]*?)<DESC>", q).group(1) 
    q_desc = re.search("<DESC>([\s\S]*)$", q).group(1)

    q_ind_clms = q_clms.split('\n\n') 
    selected_q_clm = q_ind_clms[0] 
    for q_ind_clm in q_ind_clms:
        if '(canceled)' in q_ind_clm:
            continue
        else:
            selected_q_clm = q_ind_clm
            break 
    selected_q_clm = clean_text(selected_q_clm)

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
    encoded_q_summ = tokenizer(q_summ_desc, return_tensors='pt', max_length=4096, padding='max_length', truncation=True).to(device) 
    with torch.no_grad():
        q_summ_final = pegasus.generate(**encoded_q_summ) 
        q_summ_final = tokenizer.batch_decode(q_summ_final)[0] 
        q_summ_final = q_summ_final.replace("</s>","").replace("<s>","") 
        
    ipcs.append(q_ipc) 
    titles.append(q_ttl) 
    first_claims.append(selected_q_clm) 
    descriptions.append(q_summ_final)
    
    ########################
    ### process positive ###
    ########################
    p_ttl = re.search("<TTL>([\s\S]*?)<IPC>", p).group(1) 
    p_ipc = re.search("<IPC>([\s\S]*?)<ABST>", p).group(1) 
    p_clms = re.search("<CLMS>([\s\S]*?)<DESC>", p).group(1)
    p_desc = re.search("<DESC>([\s\S]*)$", p).group(1)
    p_ind_clms = p_clms.split("\n\n") 
    selected_p_clm = p_ind_clms[0] 
    for p_ind_clm in p_ind_clms:
        if '(canceled)' in p_ind_clm:
            continue
        else:
            selected_p_clm = p_ind_clm
            break 
    selected_p_clm = clean_text(selected_p_clm) 
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
    encoded_p_summ = tokenizer(p_summ_desc, return_tensors='pt', max_length=4096, padding='max_length', truncation=True).to(device) 
    with torch.no_grad():
        p_summ_final = pegasus.generate(**encoded_p_summ) 
        p_summ_final = tokenizer.batch_decode(p_summ_final)[0] 
        p_summ_final = p_summ_final.replace("</s>","").replace("<s>","")  
    ipcs.append(p_ipc) 
    titles.append(p_ttl) 
    first_claims.append(selected_p_clm) 
    descriptions.append(p_summ_final) 
    
    ########################
    ### process negative ###
    ########################
    n_ttl = re.search("<TTL>([\s\S]*?)<IPC>", n).group(1) 
    n_ipc = re.search("<IPC>([\s\S]*?)<ABST>", n).group(1) 
    n_clms = re.search("<CLMS>([\s\S]*?)<DESC>", n).group(1)
    n_desc = re.search("<DESC>([\s\S]*)$", n).group(1)
    n_ind_clms = n_clms.split("\n\n") 
    selected_n_clm = n_ind_clms[0] 
    for n_ind_clm in n_ind_clms:
        if '(canceled)' in n_ind_clm:
            continue
        else:
            selected_n_clm = n_ind_clm
            break 
    selected_n_clm = clean_text(selected_n_clm) 
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
    encoded_n_summ = tokenizer(n_summ_desc, return_tensors='pt', max_length=4096, padding='max_length', truncation=True).to(device) 
    with torch.no_grad():
        n_summ_final = pegasus.generate(**encoded_n_summ) 
        n_summ_final = tokenizer.batch_decode(n_summ_final)[0] 
        n_summ_final = n_summ_final.replace("</s>","").replace("<s>","")  
    ipcs.append(n_ipc) 
    titles.append(n_ttl) 
    first_claims.append(selected_n_clm) 
    descriptions.append(n_summ_final) 
    
samples = pd.DataFrame() 
samples['IPC'] = ipcs 
samples['Titles'] = titles 
samples['First Claim'] = first_claims 
samples['Summarized Description'] = descriptions 

samples.to_csv("samples_100.csv", index=False) 
