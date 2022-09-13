"""
Evaluate document ranking model

The model supposed to follow pytorch-lightning-template and have `predict_step`
method which returns embedding of documents.

Author: YongWook Ha
"""
import sys
from pathlib import Path
import shutil

import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import BasePredictionWriter
import torch
from torch.utils.data import Dataset, DataLoader

import faiss
# pylint: disable=no-value-for-parameter
import pandas as pd
import numpy as np
from tqdm import tqdm

from tal_model import PatentSBERTa, Doc_ranker
from tal_data_utils import TripletData, custom_collate

class PatentDocument(Dataset):
    """Patent document as txt file"""
    def __init__(self, root: Path, is_debug=False):
        super().__init__()
        self.data = []
        if is_debug:
            with (data_path / "test_triplet.csv").open("r", encoding="utf8") as f:
                for i, triplet in enumerate(f):
                    if i >= 100000: break  # pylint: disable=multiple-statements
                    query, positive, negative = triplet.strip().split(",")
                    self.data.append(root / f"{query}.txt")
                    self.data.append(root / f"{positive}.txt")
                    self.data.append(root / f"{negative}.txt")
        else:
            for fn in root.glob("*.txt"):
                self.data.append(fn)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class Collate:
    """Read txt file"""
    def __init__(self):
        pass

    def load_file(self, fn: Path):
        """Load file and read contents"""
        with fn.open("r", encoding="utf8") as f:
            content = f.read()
        return content

    def __call__(self, batch):
        return list(map(self.load_file, batch))

class CustomWriter(BasePredictionWriter):
    def __init__(self, output_dir: str, write_interval: str):
        super().__init__(write_interval)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def write_on_batch_end(self, trainer, pl_module, prediction,
                           batch_indices: list, batch, batch_idx: int,
                           dataloader_idx: int):

        to_save = dict(zip(batch_indices, [pred.cpu() for pred in prediction]))
        batch_idx = str(batch_idx).zfill(9)
        idx = 0
        while (self.output_dir / f"batch_idx-{batch_idx}_{idx}.pt").exists():
            idx += 1
        torch.save(to_save, self.output_dir / f"batch_idx-{batch_idx}_{idx}.pt")

    def write_on_epoch_end(self, trainer, pl_module,
                           predictions: list, batch_indices: list):
        result = {}
        for pt in self.output_dir.glob("batch_idx-*.pt"):
            data = torch.load(pt)
            result.update(data)
        torch.save(result, self.output_dir / "predictions.pt")


if __name__ == "__main__":
    debug = False
    cl = True

    # Load model
    data_path = Path("/data/engines/sentence_ranker/load/FGH_spec_ind_claim_triplet_v1.4.1s/")  # ("/data/training_data/FGH/FGH_ind_claim_triplet_v1.0.0s")
    model_pt_path = Path("/pretrained/jjl/tanalysis/ipc_title_firstclaims_epoch_2_steps_6000_val_loss_0.13378801833644857.pt")
    emb_dim = 768
    params = {'checkpoint': model_pt_path,
              'from_pretrained_tok': "tanapatentlm/patentdeberta_base_spec_1024_pwi", # AI-Growth-Lab/PatentSBERTa",
              'from_pretrained_model': "tanapatentlm/patentdeberta_base_spec_1024_pwi"}  # "AI-Growth-Lab/PatentSBERTa"}
    output_dir = Path("../load_model/predictions")
    

    #model = PatentSBERTa(params, is_train=False).eval()
    dataset = TripletData(data_path, debug)
    print(len(dataset))
    emb_dict = torch.load(output_dir / "predictions.pt", map_location='cpu')
    
    for k,v in emb_dict.items():
        print(v, v.shape, v.dtype)
        print(k)
        print('=' * 30)
        break
    print(f'sample: {dataset[0]}')
    fn2id = {fn[0].stem: idx for idx, fn in enumerate(dataset)}

    # Index embedding to faiss
    if cl:
        # if use contrastive loss, you should use cosine similarity on faiss.
        index = faiss.IndexIDMap2(faiss.IndexFlatIP(emb_dim))
    else:
        index = faiss.IndexIDMap2(faiss.IndexFlatL2(emb_dim))
    if cl:
        faiss.normalize_L2(torch.stack(list(emb_dict.values()), dim=0).numpy())
    emb_dict_values = torch.stack(list(emb_dict.values()), dim=0).numpy()
    index.add_with_ids(emb_dict_values,
                       np.array(list(emb_dict.keys())))
    index.nprobe = 64

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
            for i, line in tqdm(enumerate(f), total=total_len, desc="calc mrr..."):
                if (debug and i >= 100000) or i > 100000: break  # pylint: disable=multiple-statements
                q, p, _ = line.strip().split(",")
                q_id, p_id = fn2id[q], fn2id[p]
                try:
                    q_emb = emb_dict[q_id]
                except KeyError:
                    # dataloader drop_last
                    continue
                distances, indices = index.search(np.expand_dims(q_emb, axis=0), 1000)
                rank = 1000
                r_rank = 0
                indices = indices[0].tolist()
                if p_id in indices:
                    rank = indices.index(p_id) + 1
                    r_rank = 1 / rank if rank <= 100 else 0
                df["query"].append(q)
                df["positive"].append(p)
                df["predict"].append(indices[1])
                df["rank"].append(rank)
                df["r_rank"].append(r_rank)
    except KeyboardInterrupt:
        print("stop calculating...")

    df = pd.DataFrame(df)
    print(df)
    total_count = df.count()['rank']
    for r in [1, 3, 5, 10, 20, 30, 50, 100, 1000]:
        # pylint: disable=cell-var-from-loop
        subset = df.apply(lambda x : x['r_rank'] if int(x['rank']) <= r else 0, axis=1)
        mrr = subset.sum()
        mrr_count = subset.astype(bool).sum()
        print(f"MRR@{r}: {mrr / total_count} / count: {mrr_count} / total: {total_count}")
    print(f"average Rank: {df['rank'].sum()/total_count}")

    torch.save(df, output_dir / 'df.pt')
