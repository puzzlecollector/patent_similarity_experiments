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
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import re
from transformers import (
    AdamW,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
import addict
import argparse
import faiss 

class TripletData(Dataset):
    """Patent document as txt file"""
    def __init__(self, root: Path, is_debug=False):
        super().__init__()
        self.data = []
        if is_debug:
            with (root / "test_triplet.csv").open("r", encoding="utf8") as f:
                for i, triplet in enumerate(f):
                    if i >= 100000: break  # pylint: disable=multiple-statements
                    query, positive, negative = triplet.strip().split(",")
                    data = []
                    data.append(root / f"{query}.txt")
                    data.append(root / f"{positive}.txt")
                    data.append(root / f"{negative}.txt")
                    self.data.append(data)
        else:
            for fn in root.glob("*.txt"):
                self.data.append([fn])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

        
data_path = Path("../storage/FGH_spec_ind_claim_triplet_v1.4.1s/")
emb_dim = 1024
output_dir = Path("../storage/DeBERTa_Large_embeddings_epochs2") 
debug = False 

### define dataloader ### 
dataset = TripletData(data_path, debug) 
emb_dict = torch.load(output_dir / "predictions.pt", map_location="cpu") 

### faiss calculation ### 
fn2id = {fn[0].stem: idx for idx, fn in enumerate(dataset)}

use_cosine_sim = True 

'''
if use_cosine_sim: 
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(emb_dim)) 
else: 
    index = faiss.IndexIDMap2(faiss.IndexFlatL2(emb_dim)) 
if use_cosine_sim: 
    faiss.normalize_L2(torch.stack(list(emb_dict.values()), dim=0).numpy()) 
emb_dict_values = torch.stack(list(emb_dict.values()), dim=0).numpy() 

res = faiss.StandardGpuResources() 

print("adding vector embeddings to index...") 
index.add_with_ids(emb_dict_values, np.array(list(emb_dict.keys()))) 

print("saving faiss index...") 
faiss.write_index(index, "DeBERTa-Large-epoch-2-index") 
''' 

if use_cosine_sim: 
    faiss.normalize_L2(torch.stack(list(emb_dict.values()), dim = 0).numpy()) 

print("reading saved index...") 

index = faiss.read_index("DeBERTa-Large-epoch-2-index") 


print("building gpu index...") 
res = faiss.StandardGpuResources() 
print(res) 
index = faiss.index_cpu_to_gpu(res, 0, index) 

index.nprobe = 64 

print("done!") 


df = {
    "query": [], 
    "positive": [], 
    "predict": [],
    "rank": [], 
    "r_rank": []
}
total_len = sum([1 for _ in (data_path / "test_triplet.csv").open("r", encoding="utf8")])
try: 
    with (data_path / "test_triplet.csv").open("r", encoding="utf8") as f:
        Q, P = [], [] 
        p_ids = [] 
        query_buck = [] 
        for i, line in tqdm(enumerate(f), total=total_len, desc="calculate mrr..."): 
            q, p, _ = line.strip().split(",")  
            q_id, p_id = fn2id[q], fn2id[p] 
            try: 
                q_emb = emb_dict[q_id]
                query_buck.append(q_emb) 
                Q.append(q) 
                P.append(p) 
                p_ids.append(p_id) 
            except KeyError: 
                continue 
            
            assert len(Q) == len(P) == len(p_ids) == len(query_buck) 
            
            if len(Q) == 10000:
                # calc faiss 
                q_embs = np.array(torch.stack(query_buck)) 
                b_distances, b_indices = index.search(q_embs, 1000) 
                rank = 1000 
                r_rank = 0 
                for indices in b_indices:
                    indices = indices.tolist() 
                    p_id = p_ids.pop(0) 
                    q = Q.pop(0) 
                    p = P.pop(0)
                    try: 
                        if p_id in indices: 
                            rank = indices.index(p_id) 
                            r_rank = 1 / rank if rank <= 1000 else 0.

                        df["query"].append(q) 
                        df["positive"].append(p) 
                        df["predict"].append(indices[1]) 
                        df["rank"].append(rank) 
                        df["r_rank"].append(r_rank)
                    except: 
                        pass 
                query_buck = []  
except KeyboardInterrupt: 
    print("stop calculating...") 
    
df = pd.DataFrame(df) 
print(df)
total_count = df.count()["rank"] 
for r in [1, 3, 5, 10, 20, 30, 50, 100, 1000]:
    subset = df.apply(lambda x : x["r_rank"] if int(x["rank"]) <= r else 0, axis=1)
    mrr = subset.sum() 
    mrr_count = subset.astype(bool).sum() 
    print(f"MRR@{r}: {mrr / total_count} / count: {mrr_count} / total: {total_count}")
print("average Rank : {}".format(df["rank"].sum() / total_count))

df.to_csv(output_dir / "df_all.csv", index=False)

