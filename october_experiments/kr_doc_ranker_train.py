import numpy as np 
import pandas as pd 
import os
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
    get_linear_schedule_with_warmup
) 
import addict 
import argparse 
import time 
import re 
import datetime  

class TripletData(Dataset): 
    def __init__(self, path): 
        super().__init__()
        self.data = [] 
        with Path(path).open("r", encoding="utf8") as f: 
            for i, triplet in enumerate(f): 
                try: 
                    query, positive, negative = triplet.strip().split(",") 
                    data = [] 
                    data.append("kr_triplet_v1.1/{}.txt".format(query)) 
                    data.append("kr_triplet_v1.1/{}.txt".format(positive))
                    data.append("kr_triplet_v1.1/{}.txt".format(negative)) 
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
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["[IPC]", "[TTL]", "[CLMS]", "[ABST]"]}) 
        self.chunk_size = 1024 
    def __call__(self, batch): 
        input_ids, attn_masks, labels = [], [], [] 
        ids = 0 
        for idx, triplet in enumerate(batch): 
            try: 
                query_txt, positive_txt, negative_txt = triplet
                with Path(query_txt).open("r", encoding="utf8") as f: 
                    q = f.read() 
                with Path(positive_txt).open("r", encoding="utf8") as f: 
                    p = f.read() 
                with Path(negative_txt).open("r", encoding="utf8") as f: 
                    n = f.read() 
                encoded_q = self.tokenizer(q, return_tensors="pt", max_length=self.chunk_size, padding="max_length", truncation=True) 
                encoded_p = self.tokenizer(p, return_tensors="pt", max_length=self.chunk_size, padding="max_length", truncation=True)  
                encoded_n = self.tokenizer(n, return_tensors="pt", max_length=self.chunk_size, padding="max_length", truncation=True) 
                
                input_ids.append(encoded_q["input_ids"]) 
                attn_masks.append(encoded_q["attention_mask"]) 
                labels.append(ids*2) 
                
                input_ids.append(encoded_p["input_ids"]) 
                attn_masks.append(encoded_p["attention_mask"]) 
                labels.append(ids*2) 

                input_ids.append(encoded_n["input_ids"]) 
                attn_masks.append(encoded_n["attention_mask"]) 
                labels.append(ids*2+1) 
                ids += 1 

            except Exception as e:
                print(e) 
                print("==="*100) 
                continue 
        input_ids = torch.stack(input_ids, dim=0).squeeze(dim=1) 
        attn_masks = torch.stack(attn_masks, dim=0).squeeze(dim=1) 
        labels = torch.tensor(labels, dtype=int) 
        return input_ids, attn_masks, labels 


class NeuralRanker(pl.LightningModule): 
    def __init__(self, hparams=dict(), plm="tanapatentlm/patent-ko-deberta"): 
        super(NeuralRanker, self).__init__() 
        self.hparams.update(hparams) 
        self.save_hyperparameters(ignore="hparams") 
        self.tokenizer = AutoTokenizer.from_pretrained(plm) 
        self.config = AutoConfig.from_pretrained(plm) 
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
        return {"val_loss":loss} 

    def validation_epoch_end(self, outputs): 
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean() 
        print(f"\nEpoch {self.current_epoch} | avg_loss: {avg_loss}\n") 

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int=0): 
        q_input_ids, q_attn_masks = batch["q"]["input_ids"], batch["q"]["attention_mask"] 
        q_emb = self(q_input_ids, q_attn_masks) 
        return q_emb 

if __name__ == "__main__": 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--setting", "--s", type=str, default="default.yaml", help="Experiment setting") 
    args = parser.parse_args(args=[]) 
    hparams = addict.Addict(dict(load_hparams_from_yaml(args.setting))) 
    train_set = TripletData("kr_triplet_train.csv") 
    valid_set = TripletData("kr_triplet_valid.csv")


    df1 = pd.read_csv("kr_triplet_train.csv") 
    df2 = pd.read_csv("kr_triplet_valid.csv") 
    print(df1.shape, df2.shape)


    collate = custom_collate() 
    train_dataloader = DataLoader(train_set, batch_size=hparams.batch_size, collate_fn=collate, shuffle=True) 
    valid_dataloader = DataLoader(valid_set, batch_size=hparams.batch_size, collate_fn=collate, shuffle=False) 
    model = NeuralRanker(hparams)  

    ckpt_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss", 
        dirpath="kr_chkpt/", 
        filename="epoch_end_checkpoints-{epoch:02d}-{val_loss:.8f}", 
        save_top_k=3, 
        mode="min", 
        save_last=True
    ) 

    device_cnt = torch.cuda.device_count() 
    trainer = pl.Trainer(gpus=device_cnt, 
                         max_epochs = hparams.epochs, 
                         strategy="ddp" if device_cnt > 1 else None,
                         callbacks=[ckpt_callback],
                         gradient_clip_val=1.0,
                         accumulate_grad_batches=10,
                         num_sanity_val_steps=20) 

    print("start training model!") 
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader) 
