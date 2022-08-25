from simcse import SimCSE
from sentence_transformers import models
from sentence_transformers import SentenceTransformer, util
import torch
import re
import pandas as pd
from tqdm import tqdm


model =  SentenceTransformer("/home/byeongil/python/SBERTA_SIMCSE_DBERTA/output/patent_for_bert_training_simcse-merge_Text_1000002022-06-22_05-41-03")

import os
from datetime import datetime
file_list=os.listdir("/data/training_data/FGH_spec_ind_claims_triplet_v0.3.1")
topK = 10
save_path="SIM_CSE_PATENT_FOR_BERT_0628"
os.makedirs(save_path,exist_ok=True)
full_df_list=[]
query_idx=0
for file in tqdm(file_list):
    query_num=file.split("_")[0]
    positive_num=file.split("_")[1]
    if len(os.listdir(save_path))>10000:
        break
    cur_file = open("/data/training_data/FGH_spec_ind_claims_triplet_v0.3.1/" + str(file), "r")  
    '''
    읽어온다음에 바로 .split('\n\n\n') 해도됨. 
    '''
    data=open("/data/training_data/FGH_spec_ind_claims_triplet_v0.3.1/" + str(file), "r").read()
    positive_doc=data.split("\n\n\n")[1].replace(".",";")
    IPC=data[data.find("<IPC>")+5:data.find("<ABST>")]
    lines = cur_file.readlines() 
    q_done, p_done = False, False
    cur_query, cur_positive, cur_negative = [],[],[] 
    for i in range(len(lines)):
        if q_done == False:
            cur_query.append(lines[i]) 
        elif p_done == False:
            cur_positive.append(lines[i]) 
        else:
            cur_negative.append(lines[i]) 
        if i < len(lines)-1 and lines[i] == '\n' and lines[i+1] == '\n':
            if q_done == False:
                q_done = True
            elif p_done == False:
                p_done = True 
    
    cur_query = ' '.join(cur_query) 
    start_idx = cur_query.find("<CLMS>") 
    end_idx = cur_query.find("<DESC>") 
    cur_query_claims = cur_query[start_idx+6:end_idx]

    cur_positive = ' '.join(cur_positive)
    title_start_idx = cur_positive.find("<TTL>") 
    title_end_idx = cur_positive.find("<IPC>") 
    cur_positive_title = cur_positive[title_start_idx+5:title_end_idx] 

    abstract_start_idx = cur_positive.find("<ABST>")
    abstract_end_idx = cur_positive.find("<CLMS>") 
    cur_positive_abstract = cur_positive[abstract_start_idx+6:abstract_end_idx] 

    claims_start_idx = cur_positive.find("<CLMS>") 
    claims_end_idx = cur_positive.find("<DESC>") 
    cur_positive_claims = cur_positive[claims_start_idx+6:claims_end_idx] 

    description_start_idx = cur_positive.find("<DESC>") 
    cur_positive_descriptions = cur_positive[description_start_idx+6:]  

    
    L = 100 # length threshold 
    
    splitted_query_claims = re.split(r'wherein|[;\n]+', cur_query_claims.replace(".",";"))
    splitted_query_claims_ = [] 
    for s in splitted_query_claims:
        if len(s) > L:
            splitted_query_claims_.append(s) 
    splitted_positives = [] 
    positive_start_idx={}
    positive_end_idx={}

    text = ''
    file_path="/data/training_data/FGH_spec_ind_claims_triplet_v0.3.1"+"/"+file
    with open(file_path, 'r') as f:
        for line in f:
            text = ' '.join([text, line.strip()])


    positive_doc=text[text.index("<TTL>",5):text.index("<TTL>",text.index("<TTL>",5)+5)]

    for split in re.split(r'wherein|[;\n]+', cur_positive_abstract.replace(".",";")):
        if len(split) > L: 
            splitted_positives.append(split) 
            positive_start_idx[split]=(positive_doc.find(split))
            positive_end_idx[split]=(positive_doc.find(split)+len(split))
    for split in re.split(r'wherein|[;\n]+', cur_positive_claims.replace(".",";")):
        if len(split) > L:
            splitted_positives.append(split) 
            positive_start_idx[split]=(positive_doc.find(split))
            positive_end_idx[split]=(positive_doc.find(split)+len(split))  
    for split in re.split(r'wherein|[;\n]+', cur_positive_descriptions.replace(".",";")):
        if len(split) > L:
            splitted_positives.append(split) 
            positive_start_idx[split]=(positive_doc.find(split))
            positive_end_idx[split]=(positive_doc.find(split)+len(split))
    d = {} 
    df_lists = [] 
    #model.build_index(splitted_positives)
    splitted_positives=list(set(splitted_positives))
    try:
        corpus_embeddings = model.encode(splitted_positives, convert_to_tensor=True,show_progress_bar=False)
    except:
        continue

    for i in range(len(splitted_query_claims_)):
        query_idx=query_idx+1
        d[splitted_query_claims_[i]] = []        
        query_embedding=model.encode(splitted_query_claims_[i],convert_to_tensor=True,show_progress_bar=False)
        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        topk=min(len(splitted_positives),10)
        top_results = torch.topk(cos_scores, k=topk)
        
        #for score, idx in zip(top_results[0], top_results[1]):
        sent_df, positive_df, score_df = [],[],[]
        idx_num_df,label_df, query_num_df, positive_num_df=[],[],[],[]
        positive_start_df=[]
        positive_end_df=[]
        ipc_df=[]
        j=0
        for score, idx in zip(top_results[0],top_results[1]):           

            sent_df.append(splitted_query_claims_[i])
            idx_num_df.append(query_idx)            
            positive_df.append(splitted_positives[idx])
            positive_start_df.append(positive_start_idx[splitted_positives[idx]])
            positive_end_df.append(positive_end_idx[splitted_positives[idx]])
            score_df.append("{}".format(score))
            label_df.append("")
            query_num_df.append(query_num)
            positive_num_df.append(positive_num)
            ipc_df.append(IPC)
            
        df = pd.DataFrame(list(zip(idx_num_df,ipc_df,sent_df, positive_df, score_df,label_df,query_num_df,positive_num_df,positive_start_df,positive_end_df)), 
                          columns=["쿼리 번호","IPC 분류","쿼리 문장", "후보 문장", "유사도","레이블","쿼리 문서 번호","Positive 문서 번호","후보 문장 start","후보 문장 end"])
        df_lists.append(df)
    try:
        full_df=pd.concat(df_lists)
        full_df_list.append(full_df)
    except:
        continue
    
    full_full_df=pd.concat(full_df_list)    
    #full_df.to_csv(save_path + str(file) +".csv", index=False,encoding="utf-8-sig")
    print(query_idx)
    if(query_idx)>10000:
        full_full_df.to_csv(save_path.replace("/","")+".csv",index=False,encoding="utf-8-sig")
        print(save_path.replace("/","")+".csv")
        break
