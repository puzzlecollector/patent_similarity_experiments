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

test_df = pd.read_csv("sentence_test_df.csv") 

test_queries = test_df['쿼리 문장'].values 
test_candidates = test_df['후보 문장'].values 
test_labels = test_df['라벨링'].values 


test_correlations, test_accuracies = [], [] 

for i in range(1, 11): 
    print("loading model {}...".format(i))  
    model = SentenceTransformer("../storage/simcse_patent_for_bert_KFOLD{}".format(i)) 
    model.max_seq_length = 128 
    eval_s1, eval_s2, eval_scores = [], [], [] 
    for j in range(len(test_queries)): 
        eval_s1.append(test_queries[j]) 
        eval_s2.append(test_candidates[j]) 
        eval_scores.append(test_labels[j]) 
    evaluator = evaluation.EmbeddingSimilarityEvaluator(eval_s1, eval_s2, eval_scores) 
    test_corr = model.evaluate(evaluator) 
    test_correlations.append(test_corr) 
    
    print("calculating accuracy") 
    cnt = 0 
    for j in tqdm(range(len(test_queries))): 
        q_emb = model.encode(test_queries[j]) 
        c_emb = model.encode(test_candidates[j]) 
        cosine_scores = util.cos_sim(q_emb, c_emb) 
        if test_labels[j] == 0 and cosine_scores < 0.5: 
            cnt += 1 
        elif test_labels[j] == 0.5 and (cosine_scores >= 0.5 and cosine_scores < 0.8): 
            cnt += 1 
        elif test_labels[j] >= 0.8 and cosine_scores >= 0.8: 
            cnt += 1 
    test_accuracy = cnt / len(test_labels) * 100.0 
    test_accuracies.append(test_accuracy) 
    
    print(test_corr, test_accuracy) 

print(np.mean(test_correlations)) 
print(np.mean(test_accuracies)) 


with open("simcse_patent_for_bert_test_correlations", "wb") as fp:   #Pickling
    pickle.dump(test_correlations, fp)
    
with open("simcse_patent_for_bert_test_accuracies", "wb") as fp:   #Pickling
    pickle.dump(test_accuracies, fp)
