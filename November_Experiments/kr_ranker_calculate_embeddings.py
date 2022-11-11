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

class TripletData(Dataset): 
    def __init__(self, path): 
        super().__init__() 
        self.data = [] 
        with Path(path).open("r", encoding="utf8") as f: 
            for i, triplet in enumerate(f):
                try:
                    query, positive, negative = triplet.strip().split(",")  
                    data = [] 
                    data.append(f"../storage/kr_triplet_v2.1/{query}.txt")  
                    data.append(f"../storage/kr_triplet_v2.1/{positive}.txt") 
                    data.append(f"../storage/kr_triplet_v2.1/{negative}.txt") 
                    self.data.append(data) 
                except: 
                    continue 
    def __getitem__(self, index): 
        return self.data[index] 
    def __len__(self): 
        return len(self.data) 
    
class custom_collate(object): 
    def __init__(self, plm="tanapatentlm/patent-ko-deberta"): 
        self.tokenizer = AutoTokenizer.from_pretrained(plm) 
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["[IPC]", "[TTL]", "[CLMS]", "[DESC]"]}) 
        self.chunk_size = 512 
    def __call__(self, batch): 
        q_input_ids, q_attn_masks = [], [] 
        p_input_ids, p_attn_masks = [], []
        n_input_ids, n_attn_masks = [], [] 
        for idx, triplet in enumerate(batch): 
            q_txt, p_txt, n_txt = triplet 
            with Path(q_txt).open("r", encoding="utf8") as f: 
                q = f.read() 
            with Path(p_txt).open("r", encoding="utf8") as f: 
                p = f.read() 
            with Path(n_txt).open("r", encoding="utf8") as f: 
                n = f.read() 
            encoded_q = self.tokenizer(q, return_tensors="pt", max_length=self.chunk_size, padding="max_length", truncation=True) 
            encoded_p = self.tokenizer(p, return_tensors="pt", max_length=self.chunk_size, padding="max_length", truncation=True) 
            encoded_n = self.tokenizer(n, return_tensors="pt", max_length=self.chunk_size, padding="max_length", truncation=True) 
            q_input_ids.append(encoded_q["input_ids"]) 
            q_attn_masks.append(encoded_q["attention_mask"]) 
            p_input_ids.append(encoded_p["input_ids"]) 
            p_attn_masks.append(encoded_p["attention_mask"]) 
            n_input_ids.append(encoded_n["input_ids"]) 
            n_attn_masks.append(encoded_n["attention_mask"]) 
        q_input_ids = torch.stack(q_input_ids, dim=0).squeeze(dim=1) 
        q_attn_masks = torch.stack(q_attn_masks, dim=0).squeeze(dim=1) 
        p_input_ids = torch.stack(p_input_ids, dim=0).squeeze(dim=1) 
        p_attn_masks = torch.stack(p_attn_masks, dim=0).squeeze(dim=1) 
        n_input_ids = torch.stack(n_input_ids, dim=0).squeeze(dim=1) 
        n_attn_masks = torch.stack(n_attn_masks, dim=0).squeeze(dim=1) 
        return q_input_ids, q_attn_masks, p_input_ids, p_attn_masks, n_input_ids, n_attn_masks 
    
test_data = TripletData("kr_test.csv")  
collate = custom_collate() 
dataloader = DataLoader(test_data, batch_size = 32, collate_fn=collate, shuffle=False) 
parser = argparse.ArgumentParser() 
parser.add_argument("--setting", "-s", type=str, default="default.yaml") 
args = parser.parse_args(args=[]) 
hparams = addict.Addict(dict(load_hparams_from_yaml(args.setting))) 


# define model structure 
class NeuralRanker(pl.LightningModule):
    def __init__(self, hparams=dict(), plm="tanapatentlm/patent-ko-deberta"):
        super(NeuralRanker, self).__init__()
        self.hparams.update(hparams)
        self.save_hyperparameters(ignore="hparams")
        self.tokenizer = AutoTokenizer.from_pretrained(plm)
        self.config = AutoConfig.from_pretrained(plm)
        # self.metric = torch.nn.TripletMarginLoss()
        # self.metric = (nn.TripletMarginWithDistanceLoss(distance_function = lambda x, y: 1.0 - F.cosine_similarity(x,y)))
        self.metric = losses.MultiSimilarityLoss()
        self.miner = miners.MultiSimilarityMiner()
        self.net = AutoModel.from_pretrained(plm)
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
        scheduler = {"scheduler":scheduler, "interval":"step", "frequency":1}
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        input_ids, attn_masks, labels = batch
        embeddings = self(input_ids, attn_masks)
        hard_pairs = self.miner(embeddings, labels)
        loss = self.metric(embeddings, labels, hard_pairs)
        self.log("train_loss", loss, batch_size=len(batch))
        return {"loss":loss}

    def validation_step(self, batch, batch_idx):
        input_ids, attn_masks, labels = batch
        embeddings = self(input_ids, attn_masks)
        loss = self.metric(embeddings, labels)
        self.log("val_loss", loss, batch_size=len(batch))
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        print(f"\nEpoch {self.current_epoch} | avg_loss: {avg_loss}\n")

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int=0):
        q_input_ids, q_attn_masks, p_input_ids, p_attn_masks, n_input_ids, n_attn_masks = batch
        q_embs = self(q_input_ids, q_attn_masks)
        p_embs = self(p_input_ids, p_attn_masks)
        n_embs = self(n_input_ids, n_attn_masks)
        return q_embs, p_embs, n_embs
    
model = NeuralRanker(hparams) 
model_pt_path = Path("../storage/Torch_metric_learning_KR_Ranker_val_end_chkpt-epoch=02-val_loss=0.24969509.ckpt") 
device = torch.device("cuda") 
checkpoint = torch.load(model_pt_path, map_location=device) 
loaded_dict = checkpoint["state_dict"]  
model.load_state_dict(loaded_dict) 
model.eval()
model.freeze()  
model.cuda()

device = torch.device("cuda") 

queries, positives, negatives = [], [], [] 

try: 
    for step, batch in enumerate(tqdm(dataloader, position=0, leave=True)):
        batch = tuple(t.to(device) for t in batch) 
        q_input_ids, q_attn_masks, p_input_ids, p_attn_masks, n_input_ids, n_attn_masks = batch 
        with torch.no_grad(): 
            q_emb = model(q_input_ids, q_attn_masks) 
            p_emb = model(p_input_ids, p_attn_masks) 
            n_emb = model(n_input_ids, n_attn_masks) 
        queries.append(q_emb.detach().cpu().numpy()) 
        positives.append(p_emb.detach().cpu().numpy()) 
        negatives.append(n_emb.detach().cpu().numpy()) 
except KeyboardInterrupt: 
    print("stop calculating...") 

print(len(queries), len(positives), len(negatives)) 

q_v, p_v, n_v = [], [], [] 

for q in tqdm(queries): 
    q_v.append(q) 
for p in tqdm(positives): 
    p_v.append(p) 
for n in tqdm(negatives): 
    n_v.append(n) 

q_v = np.concatenate(q_v, axis=0) 
p_v = np.concatenate(p_v, axis=0) 
n_v = np.concatenate(n_v, axis=0) 

print(q_v.shape, p_v.shape, n_v.shape) 

candidate = np.concatenate([p_v, n_v], axis=0) 

print(candidate.shape) 

embeddings = {
    "query": q_v, 
    "candidate": candidate
} 

print("saving embeddings...") 
with open("DeBERTa_KR_embeddings.pkl", "wb") as f: 
    pickle.dump(embeddings, f) 
    
