import os
import pandas as pd
import numpy as np
from sentence_transformers import models
from sentence_transformers import SentenceTransformer, util
import torch
import re

files = os.listdir("../storage/FGH_spec_ind_claims_triplet_v0.3.1")
df = pd.read_excel("라벨링세트_0821.xlsx")
positive_ids = np.unique(df['Positive 문서 번호'].values)
print(len(positive_ids))


from tqdm import tqdm

positive_docs = {}

for i in range(len(positive_ids)):
    positive_docs[positive_ids[i]] = ""


cnt = 0
for i in tqdm(range(len(positive_ids))):
    found = False
    for j in range(len(files)):
        if str(positive_ids[i]) in str(files[j]):
            with open("../storage/FGH_spec_ind_claims_triplet_v0.3.1/" + files[j], "r") as f:
                data = f.read()
            q,p,n = data.split("\n\n\n")
            positive_docs[positive_ids[i]] = p
            found = True
            break
    if found == False:
        cnt += 1

print("not found = {}".format(cnt))

chk_new_sent = {}

for i in range(len(positive_ids)):
    chk_new_sent[positive_ids[i]]=[]


full_candidates = df["후보 문장"].values
full_positive_ids = df["Positive 문서 번호"].values

for i in tqdm(range(len(full_positive_ids))):
    chk_new_sent[full_positive_ids[i]].append(full_candidates[i])


q_dict = {}
queries = df['쿼리 문장'].values
for i in range(len(positive_ids)):
    q_dict[positive_ids[i]] = ""

queries = df['쿼리 문장'].values

for i in tqdm(range(len(full_positive_ids))):
    q_dict[full_positive_ids[i]] = queries[i]


query_ids = {}
for i in range(len(positive_ids)):
    query_ids[positive_ids[i]] = ""

query_id_nums = df['쿼리 문서 번호'].values
for i in tqdm(range(len(full_positive_ids))):
    query_ids[full_positive_ids[i]] = query_id_nums[i]


model = SentenceTransformer("../storage/simcse_DEBERTA_KFOLD2")
df_lists = []

for j in tqdm(range(len(positive_ids)), position=0, leave=True):
    positive_doc = positive_docs[positive_ids[j]]
    positive_doc = positive_doc.replace(".",";")

    p_ttl = re.search("<TTL>([\s\S]*?)<IPC>", positive_doc).group(1)
    p_ipc = re.search("<IPC>([\s\S]*?)<ABST>", positive_doc).group(1)
    p_abst = re.search("<ABST>([\s\S]*?)<CLMS>", positive_doc).group(1)
    p_clms = re.search("<CLMS>([\s\S]*?)<DESC>", positive_doc).group(1)
    p_desc = re.search("<DESC>([\s\S]*)$", positive_doc).group(1)

    query = q_dict[positive_ids[j]]

    splitted_positives = []
    positive_start_idx={}
    positive_end_idx={}

    L = 100 # length threshold

    for split in re.split(r'wherein|[;\n]+', p_abst.replace(".",";")):
        if len(split) > L:
            splitted_positives.append(split)
            positive_start_idx[split]=(positive_doc.find(split))
            positive_end_idx[split]=(positive_doc.find(split)+len(split))

    for split in re.split(r'wherein|[;\n]+', p_clms.replace(".",";")):
        if len(split) > L:
            splitted_positives.append(split)
            positive_start_idx[split]=(positive_doc.find(split))
            positive_end_idx[split]=(positive_doc.find(split)+len(split))

    for split in re.split(r'wherein|[;\n]+', p_desc.replace(".",";")):
        if len(split) > L:
            splitted_positives.append(split)
            positive_start_idx[split]=(positive_doc.find(split))
            positive_end_idx[split]=(positive_doc.find(split)+len(split))


    d = {}
    splitted_positives=list(set(splitted_positives))

    try:
        corpus_embeddings = model.encode(splitted_positives, convert_to_tensor=True, show_progress_bar=False)
    except Exception as e:
        print(e)
        continue

    query_embedding = model.encode(query, convert_to_tensor=True, show_progress_bar=False)
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]

    topk = 20
    top_results = torch.topk(cos_scores, k=topk)

    sent_df, positive_df, score_df = [], [], []
    idx_num_df, label_df, query_num_df, positive_num_df = [], [], [], []
    positive_start_df, positive_end_df = [], []
    ipc_df = []

    for score, idx in zip(top_results[0], top_results[1]):
        if splitted_positives[idx] in chk_new_sent[positive_ids[j]]:
            continue
        sent_df.append(query)
        idx_num_df.append(j+1)
        positive_df.append(splitted_positives[idx])
        positive_start_df.append(positive_start_idx[splitted_positives[idx]])
        positive_end_df.append(positive_end_idx[splitted_positives[idx]])
        score_df.append("{}".format(score))
        label_df.append("")
        query_num_df.append(query_ids[positive_ids[j]])
        positive_num_df.append(positive_ids[j])
        ipc_df.append(p_ipc)

    df = pd.DataFrame(list(zip(idx_num_df,ipc_df,sent_df, positive_df, score_df,label_df,query_num_df,positive_num_df,positive_start_df,positive_end_df)),
                          columns=["쿼리 번호","IPC 분류","쿼리 문장", "후보 문장", "유사도","레이블","쿼리 문서 번호","Positive 문서 번호","후보 문장 start","후보 문장 end"])
    df_lists.append(df)

full_df = pd.concat(df_lists)
full_df.to_csv("simcse_DEBERTA_KFOLD2_sentences.csv", index=False)
