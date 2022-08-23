import numpy as np 
import pandas as pd 
from sentence_transformers import SentenceTransformer, InputExample, losses, util, evaluation
import seaborn as sns 
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedGroupKFold
import torch 
from torch.utils.data import DataLoader 
import math 
from datetime import datetime 
from transformers import AutoTokenizer, AutoModel 
from tqdm import tqdm 
import pickle 

df = pd.read_excel("0814_라벨링세트_6주차_병합.xlsx") 

df = df.loc[df['라벨링'].notnull(), ['쿼리 번호', '쿼리 문장', '후보 문장', '라벨링']] 
df = df.dropna() 
labels_fixed = [] 
labels = df['라벨링'].values 

for i in range(len(labels)): 
    if labels[i] == 0.1:
        labels_fixed.append(1.0) 
    elif labels[i] not in [0, 0.5, 0.8, 1.0]:
        labels_fixed.append(None) 
    else:
        labels_fixed.append(labels[i])  
        
df['라벨링'] = labels_fixed 
df = df.dropna() 

queries = df['쿼리 문장'].values 
candidates = df['후보 문장'].values 
labels = df['라벨링'].values 
query_no = df['쿼리 번호'].values 

cat_labels = [] 
for i in range(len(labels)): 
    if labels[i] == 0.0: 
        cat_labels.append(0) 
    elif labels[i] == 0.5: 
        cat_labels.append(1) 
    elif labels[i] == 0.8: 
        cat_labels.append(2) 
    elif labels[i] == 1.0: 
        cat_labels.append(3) 
    
val_correlations, val_accuracies = [], [] 

cv = StratifiedGroupKFold(n_splits=10) 
for idx, (train_idxs, val_idxs) in enumerate(cv.split(queries, cat_labels, query_no)): 
    print("========== KFOLD {} ==========".format(idx+1))
    train_queries, val_queries = queries[train_idxs], queries[val_idxs] 
    train_candidates, val_candidates = candidates[train_idxs], candidates[val_idxs] 
    train_labels, val_labels = labels[train_idxs], labels[val_idxs] 
    
    model = SentenceTransformer("../storage/dcpcse_patent_for_bert") 
    model.max_seq_length = 128 
    num_epochs = 1 
    train_examples = [] 
    for i in range(len(train_queries)):
        train_examples.append(InputExample(texts=[train_queries[i], train_candidates[i]], label=float(train_labels[i])))
    eval_s1, eval_s2, eval_scores = [], [], [] 
    for i in range(len(val_queries)): 
        eval_s1.append(val_queries[i]) 
        eval_s2.append(val_candidates[i]) 
        eval_scores.append(val_labels[i]) 
    train_dataloader = DataLoader(train_examples[:1000], shuffle=True, batch_size=64) 
    train_loss = losses.CosineSimilarityLoss(model=model) 
    evaluator = evaluation.EmbeddingSimilarityEvaluator(eval_s1, eval_s2, eval_scores) 
    warmup_steps = math.ceil(len(train_dataloader)*num_epochs*0.1) 
    model.fit(train_objectives=[(train_dataloader, train_loss)], 
              evaluator=evaluator, 
              epochs=num_epochs, 
              evaluation_steps=-1,
              warmup_steps=warmup_steps, 
              output_path="../storage/dcpcse_patent_for_bert_KFOLD{}".format(idx+1)) 
    
    # calculate val loss and val accuracy 
    print("load best model") 
    best_model = SentenceTransformer("../storage/dcpcse_patent_for_bert_KFOLD{}".format(idx+1)) 
    val_corr = best_model.evaluate(evaluator) 
    val_correlations.append(val_corr) 
    
    cnt = 0 
    for i in tqdm(range(len(val_queries))): 
        q_emb = best_model.encode(val_queries[i]) 
        c_emb = best_model.encode(val_candidates[i]) 
        cosine_scores = util.cos_sim(q_emb, c_emb) 
        if val_labels[i] == 0 and cosine_scores < 0.5: 
            cnt += 1 
        elif val_labels[i] == 0.5 and (cosine_scores >= 0.5 and cosine_scores < 0.8): 
            cnt += 1 
        elif val_labels[i] >= 0.8 and cosine_scores >= 0.8: 
            cnt += 1 
    val_accuracy = cnt / len(val_labels) * 100.0 
    val_accuracies.append(val_accuracy) 
    
    print("val correlation = {}".format(val_corr)) 
    print("val accuracy = {}".format(val_accuracy)) 
    
    
print(np.mean(val_correlations))
print(np.mean(val_accuracies)) 

with open("dcpcse_val_correlations", "wb") as fp:   #Pickling
    pickle.dump(val_correlations, fp)
    
with open("dcpcse_val_accuracies", "wb") as fp:   #Pickling
    pickle.dump(val_accuracies, fp)

    
print("load saved arrays") 
with open("dcpcse_val_correlations", "rb") as fp:   # Unpickling
    val_corr = pickle.load(fp)
with open("dcpcse_val_accuracies", "rb") as fp: 
    val_acc = pickle.load(fp) 
    
print(val_corr) 
print(val_acc) 
