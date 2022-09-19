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
import pickle 

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
model_pt_path = Path("epcoh_end_checkpoints-epoch=00-val_loss=0.20442404.ckpt")
emb_dim = 1024
output_dir = Path("../storage/DeBERTa_Large_embeddings")
debug = False

dataset = TripletData(data_path, debug)
collate = custom_collate(is_debug=False)
dataloader = DataLoader(dataset, batch_size=64, collate_fn=collate, shuffle=False)

result = {}
for pt in tqdm(output_dir.glob("batcdh_idx-*.pt"), desc="gathering predictions"):
    data = torch.load(pt, map_location="cpu")
    result.update(data)

print(len(result))

emb_dict = torch.load(output_dir / "predictions.pt", map_location="cpu")
fn2id = {fn[0].stem: idx for idx, fn in enumerate(dataset)}

use_cosine_sim = True

if use_cosine_sim:
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(emb_dim))
else:
    index = faiss.IndexIDMap2(faiss.IndexFlatL2(emb_dim))

if use_cosine_sim:
    faiss.normalize_L2(torch.stack(list(emb_dict.values()), dim=0).numpy())

emb_dict_values = torch.stack(list(emb_dict.values()), dim=0).numpy()
index.add_with_ids(emb_dict_values, np.array(list(emb_dict.keys())))

index.nprob = 64
df = {
    "query": [],
    "positive": [],
    "predict": [],
    "rank": [],
    "r_rank": []
}

total_len = sum([1 for _ in (data_path / "test_triplet.csv").open("r", encoding="utf8")])

instances = []

try:
    with (data_path / "test_triplet.csv").open("r", encoding="utf8") as f:
        for i, line in tqdm(enumerate(f), total=total_len, desc="calculating mrr..."):
            q, p, _ = line.strip().split(",")
            q_id, p_id = fn2id[q], fn2id[p]
            try:
                q_emb = emb_dict[q_id]
            except KeyError:
                continue
            distances, indices = index.search(np.expand_dims(q_emb, axis=0), 1000)
            rank = 1000
            r_rank = 0
            indices = indices[0].tolist()
            if p_id in indices:
                rank = indices.index(p_id) + 1
                r_rank = 1 / rank if rank <= 1000 else 0.
            if rank <= 5:
                instances.append((i, q, rank, indices[1:6])) # row index in test.csv, query, rank, top 5 candidates
except KeyboardInterrupt:
    print("stop calculating...")

print(instances) 

with open('instances.pkl','wb') as f:
  pickle.dump(instances, f) 
