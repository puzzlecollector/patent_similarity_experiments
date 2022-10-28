import numpy as np 
import pandas as pd 
import os 
from tqdm.auto import tqdm 
from transformers import (
    AdamW, 
    AutoConfig, 
    AutoModel, 
    AutoTokenizer, 
    get_linear_schedule_with_warmup
)
import torch 
import torch.nn.functional as F 
import torch.nn as nn 
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler, IterableDataset 
import math 
import time 
import datetime 
import re
from pathlib import Path
import pytorch_lightning as pl 
from pytorch_lightning.strategies.ddp import DDPStrategy 
from pytorch_lightning.callbacks import BasePredictionWriter 
from pytorch_lightning.core.saving import load_hparams_from_yaml, update_hparams 
import torch 
from torch.utils.data import Dataset, DataLoader 
from typing import List 
import addict 
import argparse

files = os.listdir("../storage/kr_triplet_v1.1") 
available_df = pd.read_csv("../storage/kr_triplet_v1.1/available_data.csv") 

hash_table = {}

for f in tqdm(files): 
    if ".csv" not in f: 
        hash_table[f[:-4]] = True 

data_path = Path("../storage/kr_triplet_v1.1/available_data.csv") 

triplets = [] 

total_len = sum([1 for _ in data_path.open("r",encoding="utf8")]) 

with (data_path).open("r", encoding="utf8") as f: Path
    for i, line in tqdm(enumerate(f), total=total_len): 
        q, p, n = line.strip().split(",") 
        if (q in hash_table.keys()) and (p in hash_table.keys()) and (n in hash_table.keys()): 
            triplets.append((q,p,n))
            
train_size = int(len(triplets) * 0.8) 
val_size = int(len(triplets) * 0.1) 

train_triplets = triplets[:train_size] 
valid_triplets = triplets[train_size:train_size+val_size]
test_triplets = triplets[train_size+val_size:] 

class TripletData(Dataset): 
    def __init__(self, arr): 
        super().__init__()
        self.arr = arr 
        self.data = []
        for i in range(len(arr)): 
            try: 
                query, positive, negative = self.arr[i] 
                data = [] 
                data.append("../storage/kr_triplet_v1.1/{}.txt".format(query)) 
                data.append("../storage/kr_triplet_v1.1/{}.txt".format(positive)) 
                data.append("../storage/kr_triplet_v1.1/{}.txt".format(negative)) 
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
        self.tokenizer.add_special_tokens({"additional_special_tokens":["[IPC]", "[TTL]", "[CLMS]", "[DESC]"]}) 
        self.chunk_size = 1024 
    def __call__(self, batch): 
        input_ids, attn_masks, labels = [], [], [] 
        for idx, triplet in enumerate(batch): 
            try: 
                query_txt, positive_txt, negative_txt = triplet 
                with Path(query_txt).open("r", encoding="utf8") as f: 
                    q = f.read() 
                with Path(positive_txt).open("r", encoding="utf8") as f: 
                    p = f.read() 
                with Path(negative_txt).open("r", encoding="utf8") as f: 
                    n = f.read() 
                
                positive_encoded = self.tokenizer(q, p, return_tensors="pt", max_length=self.chunk_size, padding="max_length", truncation=True) 
                negative_encoded = self.tokenizer(q, n, return_tensors="pt", max_length=self.chunk_size, padding="max_length", truncation=True) 
                
                input_ids.append(positive_encoded["input_ids"]) 
                attn_masks.append(positive_encoded["attention_mask"]) 
                labels.append(1) 
                
                input_ids.append(negative_encoded["input_ids"]) 
                attn_masks.append(negative_encoded["attention_mask"]) 
                labels.append(0) 
            except Exception as e:
                print(e)
                continue 
        input_ids = torch.stack(input_ids, dim=0).squeeze(dim=1) 
        attn_masks = torch.stack(attn_masks, dim=0).squeeze(dim=1) 
        labels = torch.tensor(labels, dtype=int) 
        return input_ids, attn_masks, labels
      
class MeanPooling(nn.Module): 
    def __init__(self): 
        super(MeanPooling, self).__init__() 
    def forward(self, last_hidden_state, attention_mask): 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float() 
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1) 
        sum_mask = input_mask_expanded.sum(1) 
        sum_mask = torch.clamp(sum_mask, min=1e-9) 
        mean_embeddings = sum_embedding / sum_mask 
        return mean_embeddings 

class MultiSampleDropout(nn.Module): 
    def __init__(self, max_dropout_rate, num_samples, classifier): 
        super(MultiSampleDropout, self).__init__() 
        self.dropout = nn.Dropout
        self.classifier = classifier 
        self.max_dropout_rate = max_dropout_rate 
        self.num_samples = num_samples
    def forward(self, out): 
        return torch.mean(torch.stack([self.classifier(self.dropout(p=self.max_dropout_rate)(out)) for _, rate in enumerate(np.linspace(0, self.max_dropout_rate, self.num_samples))], dim=0), dim=0) 
    
    
class NeuralClf(pl.LightningModule): 
    def __init__(self, hparams=dict(), plm="tanapatentlm/patent-ko-deberta"): 
        super(NeuralClf, self).__init__()
        self.config = AutoConfig.from_pretrained(plm) 
        self.model = AutoModel.from_pretrained(plm, config=self.config) 
        self.tokenizer = AutoTokenizer.from_pretrained(plm) 
        self.mean_pooler = MeanPooling() 
        self.fc = nn.Linear(self.config.hidden_size, 2) 
        self._init_weights(self.fc) 
        self.multi_dropout = MultiSampleDropout(0.2, 8, self.fc) 
        self.metric = nn.CrossEntropyLoss() 
        
    def _init_weights(self, module): 
        if isinstance(module, nn.Linear): 
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range) 
            if module.bias is not None: 
                module.bias.data.zero_()  
        
    def forward(self, input_ids, attn_masks): 
        x = self.model(input_ids, attn_masks)[0] 
        x = self.mean_pooler(x) 
        x = self.multi_dropout(x) 
        return x 

    def configure_optimizers(self): 
        optimizer = torch.optim.AdamW(self.parameters(), 
                                      lr=float(2e-5), 
                                      weight_decay=float(0.0), 
                                      eps=float(1e-8)) 
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps = 300, 
            num_training_steps = self.trainer.estimated_stepping_batches, 
        ) 
        scheduler  = {"scheduler": scheduler, "interval": "step", "frequency":1} 
        return [optimizer], [scheduler] 

    def training_step(self, batch, batch_idx): 
        input_ids, attn_masks, labels = batch 
        outputs = self(input_ids, attn_masks) 
        loss = self.metric(outputs, labels) 
        self.log("train_loss", loss, batch_size=len(batch)) 
        return {"loss": loss} 

    def validation_step(self, batch, batch_idx):
        input_ids, attn_masks, labels = batch 
        outputs = self(input_ids, attn_masks) 
        loss = self.metric(outputs, labels) 
        self.log("val_loss", loss, batch_size=len(batch)) 
        return {"val_loss": loss} 

    def validation_epoch_end(self, outputs): 
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean() 
        print(f"\nEpoch {self.current_epoch} | avg_loss:{avg_loss}\n") 
    
    def predict_step(self, batch, batch_idx: int, dataloader_idx: int=0):
        input_ids, attn_masks = batch 
        logits = self(input_ids, attn_masks) 
        return logits 

if __name__ == "__main__": 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--setting", "-s", type=str, default="clf_configs.yaml", help="Experiment settings") 
    args = parser.parse_args(args=[]) 
    hparams = addict.Addict(dict(load_hparams_from_yaml(args.setting))) 
    
    train_set = TripletData(train_triplets) 
    valid_set = TripletData(valid_triplets) 
    collate = custom_collate()
    
    train_dataloader = DataLoader(train_set, batch_size=2, collate_fn=collate, shuffle=True) 
    valid_dataloader = DataLoader(valid_set, batch_size=2, collate_fn=collate, shuffle=False)
    
    model = NeuralClf(hparams) 
    
    ckpt_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss", 
        dirpath="kr_clf_chkpts/",
        filename="clf_{epoch:02d}_{val_loss:.8f}", 
        save_top_k=3,
        mode="min",
        save_last=True
    ) 
    
    device_cnt = torch.cuda.device_count() 
    trainer = pl.Trainer(gpus=device_cnt, 
                         max_epochs=5,
                         strategy="ddp" if device_cnt > 1 else None, 
                         callbacks=[ckpt_callback], 
                         gradient_clip_val=1.0,
                         accumulate_grad_batches=10,
                         num_sanity_val_steps=20) 
    print("Start training model!")
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader) 
