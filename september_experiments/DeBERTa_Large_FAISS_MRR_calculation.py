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


class custom_collate(object): 
    def __init__(self, plm="tanapatentlm/patentdeberta_base_spec_1024_pwi", is_debug=False):
        self.is_debug = is_debug 
        self.tokenizer = AutoTokenizer.from_pretrained(plm)
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["[IPC]", "[TTL]", "[CLMS]", "[ABST]"]})
        self.chunk_size = 512
    
    def load_file(self, fn_list: List[Path]): 
        triplet = [] 
        for fn in fn_list: 
            with fn.open("r", encoding="utf8") as f: 
                content = f.read() 
                triplet.append(content) 
        return triplet 

    def clean_text(self, t):
        x = re.sub("\d+","",t)
        x = x.replace("\n"," ")
        x = x.strip()
        return x

    def encode_document(self, doc_list): 
        text_input = [] 
        for doc in doc_list: 
            transformed_text = self.transform_document(doc) 
            text_input.append(transformed_text) 
        encoded_doc = self.tokenizer(text_input, 
                                     return_tensors="pt", 
                                     max_length=self.chunk_size, 
                                     padding="max_length", 
                                     truncation=True) 
        return encoded_doc 

    def transform_document(self, doc): 
        ttl = re.search("<TTL>([\s\S]*?)<IPC>", doc).group(1)
        ttl = ttl.lower()
        ipc = re.search("<IPC>([\s\S]*?)<ABST>", doc).group(1)
        ipc = ipc[:3]
        clms = re.search("<CLMS>([\s\S]*?)<DESC>", doc).group(1)
        ind_clms = clms.split("\n\n")
        clean_ind_clms = []
        for ind_clm in ind_clms:
            if "(canceled)" in ind_clm:
                continue
            else:
                clean_ind_clms.append(self.clean_text(ind_clm))
        text_input = "[IPC]" + ipc + "[TTL]" + ttl
        for i in range(len(clean_ind_clms)):
            text_input += "[CLMS]" + clean_ind_clms[i] 
        return text_input 
    
    def __call__(self, batch): 
        if self.is_debug: 
            triplet = zip(*list(map(self.load_file, batch)))
            q,p,n = triplet 
            assert len(q) == len(p) == len(n) 
            encoded_q = self.encode_document(q) 
            encoded_p = self.encode_document(p) 
            encoded_n = self.encode_document(n) 
            return {"q": encoded_q, 
                    "p": encoded_p, 
                    "n": encoded_n} 
        else:
            q = list(map(self.load_file, batch)) 
            q = [qq[0] for qq in q] 
            encoded_q = self.encode_document(q) 
            return {"q": encoded_q} 

        
class NeuralRanker(pl.LightningModule):
    def __init__(self,
                 hparams=dict(),
                 plm="tanapatentlm/patentdeberta_base_spec_1024_pwi",
                 is_train=True,
                 loss_type="ContrastiveLoss",
                 use_miner=True):
        super(NeuralRanker, self).__init__()
        self.hparams.update(hparams)
        self.save_hyperparameters(ignore="hparams")
        self.tokenizer = AutoTokenizer.from_pretrained(plm)
        self.config = AutoConfig.from_pretrained(plm)
        self.is_train = is_train
        self.loss_type = loss_type
        self.use_miner = use_miner

        if plm == "tanapatentlm/patentdeberta_base_spec_1024_pwi":
            print("initialize PLM from previous checkpoint trained on FGH_v0.3.1")
            self.net = AutoModel.from_pretrained(plm)
            state_dict = torch.load(hparams["checkpoint"], map_location=self.device)
            new_weights = self.net.state_dict()
            old_weights = list(state_dict.items())
            i = 0
            for k, _ in new_weights.items():
                new_weights[k] = old_weights[i][1]
                i += 1
            self.net.load_state_dict(new_weights)
        else:
            self.net = AutoModel.from_pretrained(plm)

        if self.is_train == False:
            self.net.eval()  # change to evaluation mode

        if self.loss_type == "ContrastiveLoss":
            self.metric = losses.ContrastiveLoss()  # default is L2 distance
            # # change cosine similarity
            # self.metric = losses.ContrastiveLoss(
            #     pos_margin=1, neg_margin=0,
            #     distance=CosineSimilarity(),
            # )
        elif self.loss_type == "TripletMarginLoss":
            self.metric = losses.TripletMarginLoss()
        elif self.loss_type == "MultiSimilarityLoss":
            self.metric = losses.MultiSimilarityLoss()

        if self.use_miner:
            self.miner = miners.MultiSimilarityMiner()

        if "additional_special_tokens" in self.hparams and self.hparams["additional_special_tokens"]:
            additional_special_tokens = self.hparams["additional_special_tokens"]
            self.tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})
            self.net.resize_token_embeddings(len(self.tokenizer))

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, input_ids, attention_mask):
        model_output = self.net(input_ids, attention_mask)
        model_output = self.mean_pooling(model_output, attention_mask)
        return model_output

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=float(self.hparams.lr),
                                      weight_decay=float(self.hparams.weight_decay),
                                      eps=float(self.hparams.adam_epsilon))
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        input_ids, attn_masks, labels = batch
        embeddings = self(input_ids, attn_masks)
        if self.use_miner:
            hard_pairs = self.miner(embeddings, labels)
            loss = self.metric(embeddings, labels, hard_pairs)
        else:
            loss = self.metric(embeddings, labels)
        self.log("train_loss", loss, batch_size=len(batch))
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        input_ids, attn_masks, labels = batch
        embeddings = self(input_ids, attn_masks)
        loss = self.metric(embeddings, labels)
        self.log("val_loss", loss, batch_size=len(batch))
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        print(f"\nEpoch {self.current_epoch} | avg_loss:{avg_loss}\n")

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int=0):
        q_input_ids, q_attn_masks = batch["q"]["input_ids"], batch["q"]["attention_mask"] 
        q_emb = self(q_input_ids, q_attn_masks) 
        return q_emb 
    
        
ckpt = "epoch_end_checkpoints-epoch=01-val_loss=0.21802041.ckpt" 

class NeuralRanker(pl.LightningModule):
    def __init__(self,
                 hparams=dict(),
                 plm="tanapatentlm/patentdeberta_base_spec_1024_pwi",
                 is_train=True,
                 loss_type="ContrastiveLoss",
                 use_miner=True):
        super(NeuralRanker, self).__init__()
        self.hparams.update(hparams)
        self.save_hyperparameters(ignore="hparams")
        self.tokenizer = AutoTokenizer.from_pretrained(plm)
        self.config = AutoConfig.from_pretrained(plm)
        self.is_train = is_train
        self.loss_type = loss_type
        self.use_miner = use_miner

        if plm == "tanapatentlm/patentdeberta_base_spec_1024_pwi":
            print("initialize PLM from previous checkpoint trained on FGH_v0.3.1")
            self.net = AutoModel.from_pretrained(plm)
            state_dict = torch.load(hparams["checkpoint"], map_location=self.device)
            new_weights = self.net.state_dict()
            old_weights = list(state_dict.items())
            i = 0
            for k, _ in new_weights.items():
                new_weights[k] = old_weights[i][1]
                i += 1
            self.net.load_state_dict(new_weights)
        else:
            self.net = AutoModel.from_pretrained(plm)

        if self.is_train == False:
            self.net.eval()  # change to evaluation mode

        if self.loss_type == "ContrastiveLoss":
            self.metric = losses.ContrastiveLoss()  # default is L2 distance
            # # change cosine similarity
            # self.metric = losses.ContrastiveLoss(
            #     pos_margin=1, neg_margin=0,
            #     distance=CosineSimilarity(),
            # )
        elif self.loss_type == "TripletMarginLoss":
            self.metric = losses.TripletMarginLoss()
        elif self.loss_type == "MultiSimilarityLoss":
            self.metric = losses.MultiSimilarityLoss()

        if self.use_miner:
            self.miner = miners.MultiSimilarityMiner()

        if "additional_special_tokens" in self.hparams and self.hparams["additional_special_tokens"]:
            additional_special_tokens = self.hparams["additional_special_tokens"]
            self.tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})
            self.net.resize_token_embeddings(len(self.tokenizer))

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, input_ids, attention_mask):
        model_output = self.net(input_ids, attention_mask)
        model_output = self.mean_pooling(model_output, attention_mask)
        return model_output

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=float(self.hparams.lr),
                                      weight_decay=float(self.hparams.weight_decay),
                                      eps=float(self.hparams.adam_epsilon))
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        input_ids, attn_masks, labels = batch
        embeddings = self(input_ids, attn_masks)
        if self.use_miner:
            hard_pairs = self.miner(embeddings, labels)
            loss = self.metric(embeddings, labels, hard_pairs)
        else:
            loss = self.metric(embeddings, labels)
        self.log("train_loss", loss, batch_size=len(batch))
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        input_ids, attn_masks, labels = batch
        embeddings = self(input_ids, attn_masks)
        loss = self.metric(embeddings, labels)
        self.log("val_loss", loss, batch_size=len(batch))
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        print(f"\nEpoch {self.current_epoch} | avg_loss:{avg_loss}\n")

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int=0):
        q_input_ids, q_attn_masks, p_input_ids, p_attn_masks, n_input_ids, n_attn_masks = zip(*batch)
        q_input_ids = torch.stack(q_input_ids).squeeze(dim=1) 
        q_attn_masks = torch.stack(q_attn_masks).squeeze(dim=1) 
        p_input_ids = torch.stack(p_input_ids).squeeze(dim=1) 
        p_attn_masks = torch.stack(p_attn_masks).squeeze(dim=1) 
        n_input_ids = torch.stack(n_input_ids).squeeze(dim=1) 
        n_attn_masks = torch.stack(n_attn_masks).squeeze(dim=1)         
        q_emb = self(q_input_ids, q_attn_masks)
        p_emb = self(p_input_ids, p_attn_masks)
        n_emb = self(n_input_ids, n_attn_masks)
        return q_emb, p_emb, n_emb

data_path = Path("../storage/FGH_spec_ind_claim_triplet_v1.4.1s/")
model_pt_path = Path("epoch_end_checkpoints-epoch=00-val_loss=0.20442404.ckpt")
emb_dim = 1024
output_dir = Path("../storage/DeBERTa_Large_embeddings")
debug = False 

### define dataloader ### 
dataset = TripletData(data_path, debug) 
collate = custom_collate(is_debug=False) 
dataloader = DataLoader(dataset, batch_size=680, collate_fn=collate, shuffle=False) 

result = {} 
for pt in tqdm(output_dir.glob("batch_idx-*.pt"), desc="gathering predictions"): 
    data = torch.load(pt, map_location="cpu") 
    result.update(data) 
print(len(result), len(dataloader))  

torch.save(result, output_dir / "predictions.pt") 
emb_dict = torch.load(output_dir / "predictions.pt", map_location="cpu") 

### faiss calculation ### 
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
        for i, line in tqdm(enumerate(f), total=total_len, desc="calculate mrr..."): 
            if debug and i >= 100000: break 
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
            df["query"].append(q) 
            df["positive"].append(p) 
            df["predict"].append(indices[1]) 
            df["rank"].append(rank) 
            df["r_rank"].append(r_rank) 
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

df.to_csv(output_dir / "df.csv", index=False)

