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
if __name__ == "__main__":
    debug = False
    # Load model
    data_path = Path("/data/engines/sentence_ranker/load/FGH_spec_ind_claim_triplet_v1.4.1s/")  # ("/data/training_data/FGH/FGH_ind_claim_triplet_v1.0.0s")
    model_pt_path = Path("/pretrained/jjl/tanalysis/ipc_title_firstclaims_epoch_2_steps_6000_val_loss_0.13378801833644857.pt")
    emb_dim = 768
    params = {'checkpoint': model_pt_path,
              'from_pretrained_tok': "tanapatentlm/patentdeberta_base_spec_1024_pwi", # AI-Growth-Lab/PatentSBERTa",
              'from_pretrained_model': "tanapatentlm/patentdeberta_base_spec_1024_pwi"}  # "AI-Growth-Lab/PatentSBERTa"}
    output_dir = Path("./predictions")
    
    df = torch.load(output_dir / 'df.pt')
    import json
    D = {}
    with open('./check_ranked_docs.json', 'w') as j:
        for i in range(len(df)):
            dicts = {"query": [], 'rank_for_positive_doc':[], "predict":[], "predict_1000":[]}
            pred = df.iloc[i, 2]
            query = df.iloc[i, 0]
            pred_1000 = df.iloc[i,3]
            rank = int(df.iloc[i, 4])
            dicts['rank_for_positive_doc'].append(rank)
            with (data_path / f"{query}.txt").open("r", encoding="utf8") as f:
                content = f.read()
                dicts['query'].append(content)
            if rank > 1000:
                pred_1000 = id2fn[pred_1000]
                with (pred_1000).open("r", encoding="utf8") as f:
                    content = f.read()
                    dicts['predict_1000'].append(content)
            P = []
            for p in pred:
                p = id2fn[p]
                try:
                    with (p).open("r", encoding="utf8") as f:
                        content = f.read()
                        dicts['predict'].append(content)
                except:
                    print('pred: ')
                    print(p)
                    pass
            D[f"{i}_query"] = dicts
        json.dump(D, j, indent=4)
