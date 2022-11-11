import numpy as np 
import pandas as pd 
import torch 
import torch.nn.functional as F 
import torch.nn as nn  
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler, IterableDataset 
from pytorch_metric_learning import miners, losses 
from pytorch_metric_learning.distances import CosineSimilarity 
import sys 
from pathlib import Path 
import shutil 
import pytorch_lightning as pl 
from pytorch_lightning.strategies.ddp import DDPStrategy 
from pytorch_lightning.callbacks import BasePredictionWriter 
from pytorch_lightning.core.saving import load_hparams_from_yaml, update_hparams 
import torch 
from torch.utils.data import Dataset, DataLoader 
from typing import List 
from transformers import *  
import addict
import argparse 
from scipy.spatial import distance 
from functools import partial 
from tqdm.contrib.concurrent import process_map
from tqdm.auto import tqdm
import pickle 
import os 

os.environ["TOKENIZERS_PARALLELISM"] = "true" 

with open("DeBERTa_KR_embeddings.pkl", "rb") as f: 
    embeddings = pickle.load(f) 
    q_v = embeddings["query"] 
    candidate = embeddings["candidate"] 

print(q_v.shape, candidate.shape) 

def get_rank(inp, candidate):
    i, q = inp 
    distances = distance.cdist([q], candidate.copy(), "cosine")[0] 
    rank = np.argsort(distances) 
    return rank[0], np.where(rank==i)[0][0] + 1 

ranks = process_map(partial(get_rank, candidate=candidate), 
                    enumerate(q_v[:1000]), 
                    total=len(q_v[:1000]), 
                    max_workers=32) 

p1, rank = zip(*ranks) 
rrank = [] 
for r in rank: 
    if r <= 1000: 
        rrank.append(1/r) 
    else: 
        rrank.append(0) 
    
print(f"MRR@1000: {np.mean(rrank)}")
print(f"average rank: {np.mean(rank)}")