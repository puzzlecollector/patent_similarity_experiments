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
from scipy.spatial import distance 
import tqdm 
from tqdm.contrib.concurrent import process_map 
from functools import partial 


dest_dir = "../storage/ipc_title_firstclaim.pkl"

print("selected dest_dir: {}".format(dest_dir))


with open(dest_dir, "rb") as f:
    embeddings = pickle.load(f) 
    q_v = embeddings['query']
    candidate = embeddings['candidate'] 
    
print(q_v.shape, candidate.shape) 

def get_rank(inp, candidate): 
    i, q = inp
    distances = distance.cdist([q], candidate.copy(), "euclidean")[0] 
    rank = np.argsort(distances) 
    return rank[0], np.where(rank==i)[0][0]+1 

ranks = process_map(partial(get_rank, candidate=candidate),
                    enumerate(q_v), 
                    total=len(q_v),
                    max_workers=32) 

p1, rank = zip(*ranks) 
result = pd.DataFrame() 
result['p1'] = p1 
result['rank'] = rank 

result['r_rank'] = 1 / result['rank']

total_count = result.count()['rank']
for i, r in enumerate([1, 3, 5, 10, 20, 30, 50, 100]):
    subset = result.apply(lambda x : x['r_rank'] if int(x['rank']) <= r else 0, axis=1)
    mrr = subset.sum()
    mrr_count = subset.astype(bool).sum()
    print(f'MRR@{r}:', mrr / total_count, '/ count:', mrr_count)

print("average rank = {}".format(result['rank'].sum() / total_count)) 
print("saving results dataframe...") 
result.to_csv("../storage/ipc_title_firstclaim.csv",index=False) 

print("done!!!!!") 