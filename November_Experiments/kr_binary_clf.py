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

class TripletData(Dataset): 
    def __init__(self, path): 
        super().__init__() 
        self.data = [] 
        with Path(path).open("r", encoding="utf8") as f: 
            for i, triplet in enumerate(f): 
                query, positive, negative = triplet.strip().split(",") 
                data = [] 
                data.append("kr_triplet_v2.1/{}.txt".format(query)) 
                data.append("kr_triplet_v2.1/{}.txt".format(positive)) 
                data.append("kr_triplet_v2.1/{}.txt".format(negative))
                self.data.append(data) 
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
        input_ids, attn_masks, labels = [], [], [] 
        for idx, triplet in enumerate(batch): 
            q_txt, p_txt, n_txt = triplet 
            with Path(q_txt).open("r", encoding="utf8") as f: 
                q = f.read() 
            with Path(p_txt).open("r", encoding="utf8") as f: 
                p = f.read() 
            with Path(n_txt).open("r", encoding="utf8") as f: 
                n = f.read() 
            encoded_p = self.tokenizer(q, p, return_tensors="pt", max_length=self.chunk_size, padding="max_length", truncation=True)
            encoded_n = self.tokenizer(q, n, return_tensors="pt", max_length=self.chunk_size, padding="max_length", truncation=True) 
            input_ids.append(encoded_p["input_ids"]) 
            attn_masks.append(encoded_p["attention_mask"]) 
            labels.append(1) # positive pair  

            input_ids.append(encoded_n["input_ids"]) 
            attn_masks.append(encoded_n["attention_mask"]) 
            labels.append(0) # negative pair 
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
        mean_embeddings = sum_embeddings / sum_mask 
        return mean_embeddings 

class MultiSampleDropout(nn.Module): 
    def __init__(self, max_dropout_rate, num_samples, classifier): 
        super(MultiSampleDropout, self).__init__() 
        self.dropout = nn.Dropout
        self.classifier = classifier 
        self.max_dropout_rate = max_dropout_rate 
        self.num_samples = num_samples 
    def forward(self, out): 
        return torch.mean(torch.stack([
            self.classifier(self.dropout(p=self.max_dropout_rate)(out))
            for _, rate in enumerate(np.linspace(0, self.max_dropout_rate, self.num_samples))
        ], dim=0), dim=0) 

class WeightedLayerPooling(nn.Module): 
    def __init__(self, num_hidden_layers, layer_start: int=4, layer_weights=None): 
        super(WeightedLayerPooling, self).__init__() 
        self.layer_start = layer_start 
        self.num_hidden_layers = num_hidden_layers 
        self.layer_weights = nn.Parameter(torch.tensor([1] * (num_hidden_layers+1 - layer_start), dtype=torch.float)) 
    def forward(self, all_hidden_states): 
        all_layer_embedding = torch.stack(list(all_hidden_states), dim=0) 
        all_layer_embedding = all_layer_embedding[self.layer_start:, :, :, :] 
        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size()) 
        weighted_average = (weight_factor * all_layer_embedding).sum(dim=0) / self.layer_weights.sum() 
        return weighted_average 

class Classifier(pl.LightningModule): 
    def __init__(self, hparams=dict(), plm="tanapatentlm/patent-ko-deberta"): 
        super(Classifier, self).__init__() 
        self.num_classes = 2 
        self.hparams.update(hparams) 
        self.save_hyperparameters(ignore="hparams") 
        self.tokenizer = AutoTokenizer.from_pretrained(plm) 
        self.config = AutoConfig.from_pretrained(plm) 
        self.net = AutoModel.from_pretrained(plm) 
        self.mean_pooling = MeanPooling() 
        self.weighted_layer_pooling = WeightedLayerPooling(self.config.num_hidden_layers, 9, None) 
        self.fc = nn.Linear(self.config.hidden_size, self.num_classes) 
        self._init_weights(self.fc) 
        self.multi_dropout = MultiSampleDropout(0.2, 8, self.fc) 
        self.metric = nn.CrossEntropyLoss() 
        if "additional_special_tokens" in self.hparams and self.hparams["additional_special_tokens"]: 
            additional_special_tokens = self.hparams["additional_special_tokens"] 
            self.tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens}) 
            self.net.resize_token_embeddings(len(self.tokenizer))
    def _init_weights(self, module): 
        if isinstance(module, nn.Linear): 
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range) 
            if module.bias is not None:
                module.bias.data.zero_() 
    def forward(self, input_ids, attention_mask): 
        x = self.net(input_ids, attention_mask, output_hidden_states=True) 
        x = self.weighted_layer_pooling(x.hidden_states) 
        x = self.mean_pooling(x, attention_mask) 
        x = self.multi_dropout(x) 
        return x 
    def configure_optimizers(self): 
        optimizer = torch.optim.AdamW(self.parameters(), lr=float(self.hparams.lr), weight_decay=float(self.hparams.weight_decay), eps=float(self.hparams.adam_epsilon)) 
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.trainer.estimated_stepping_batches) 
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1} 
        return [optimizer], [scheduler] 
    def training_step(self, batch, batch_idx): 
        input_ids, attn_masks, labels = batch 
        output = self(input_ids, attn_masks) 
        loss = self.metric(output, labels) 
        self.log("train_loss", loss, batch_size=len(batch)) 
        return {"loss": loss} 
    def validation_step(self, batch, batch_idx): 
        input_ids, attn_masks, labels = batch 
        output = self(input_ids, attn_masks) 
        loss = self.metric(output, labels) 
        self.log("val_loss", loss, batch_size=len(batch)) 
        return {"val_loss": loss} 
    def validation_epoch_end(self, outputs): 
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        print(f"\nEpoch {self.current_epoch} | avg_loss: {avg_loss}\n") 
    def predict_step(self, batch, batch_idx: int, dataloader_idx: int=0): 
        sample_input_ids, sample_attn_masks = batch 
        return self(sample_input_ids, sample_attn_masks) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--setting", "-s", type=str, default="default.yaml") 
    args = parser.parse_args(args=[])
    hparams = addict.Addict(dict(load_hparams_from_yaml(args.setting))) 
    
    train_set = TripletData("train_filter.csv") 
    valid_set = TripletData("val_filter.csv") 
    collate = custom_collate() 
    train_dataloader = DataLoader(train_set, batch_size=hparams.batch_size, collate_fn=collate, shuffle=True) 
    valid_dataloader = DataLoader(valid_set, batch_size=hparams.batch_size, collate_fn=collate, shuffle=False) 
    
    model = Classifier(hparams) 
    ckpt_callback = pl.callbacks.ModelCheckpoint(
        monitor = "val_loss", 
        dirpath="clf_checkpoints/", 
        filename="kr_clf_{epoch:02d}_{val_loss:.8f}", 
        save_top_k = 3, 
        mode = "min", 
        save_last=True
    ) 
    device_cnt = torch.cuda.device_count() 
    trainer = pl.Trainer(gpus=device_cnt, max_epochs=hparams.epochs, strategy="ddp", callbacks=[ckpt_callback], gradient_clip_val=1.0, accumulate_grad_batches=4, num_sanity_val_steps=30) 
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader) 
