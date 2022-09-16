import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler, IterableDatset
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

class TripletData(Dataset):
    def __init__(self, path):
        super().__init__()
        self.data = []
        with Path(path).open("r", encoding="utf8") as f:
            for i, triplet in enumerate(f):
                try:
                    query, positive, negative = triplet.strip().split(",")
                    data = []
                    data.append("./FGH_spec_ind_claim_triplet_v1.4.1s/{}.txt".format(query))
                    data.append("./FGH_spec_ind_claim_triplet_v1.4.1s/{}.txt".format(positive))
                    data.append("./FGH_spec_ind_claim_triplet_v1.4.1s/{}.txt".format(negative))
                    self.data.append(data)
                except:
                    continue

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class custom_collate(object):
    def __init__(self, plm="tanapatentlm/patentdeberta_large_spec_128_pwi"):
        self.tokenizer = AutoTokenizer.from_pretrained(plm)
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["[IPC]", "[TTL]", "[CLMS]", "[ABST]"]})
        self.chunk_size = 512

    def clean_text(self, t):
        x = re.sub("\d+","",t)
        x = x.replace("\n"," ")
        x = x.strip()
        return x

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
                q_ttl = re.search("<TTL>([\s\S]*?)<IPC>", q).group(1)
                q_ttl = q_ttl.lower()
                q_ipc = re.search("<IPC>([\s\S]*?)<ABST>", q).group(1)
                q_ipc = q_ipc[:3]
                q_clms = re.search("<CLMS>([\s\S]*?)<DESC>", q).group(1)
                q_ind_clms = q_clms.split("\n\n")
                q_clean_ind_clms = []
                for q_ind_clm in q_ind_clms:
                    if "(canceled)" in q_ind_clm:
                        continue
                    else:
                        q_clean_ind_clms.append(self.clean_text(q_ind_clm))
                q_text_input = "[IPC]" + q_ipc + "[TTL]" + q_ttl
                for i in range(len(q_clean_ind_clms)):
                    q_text_input += "[CLMS]" + q_clean_ind_clms[i]
                encoded_q = self.tokenizer(q_text_input, return_tensors="pt", max_length=self.chunk_size, padding="max_length", truncation=True)

                p_ttl = re.search("<TTL>([\s\S]*?)<IPC>", p).group(1)
                p_ttl = p_ttl.lower()
                p_ipc = re.search("<IPC>([\s\S]*?)<ABST>", p).group(1)
                p_ipc = p_ipc[:3]
                p_clms = re.search("<CLMS>([\s\S]*?)<DESC>", p).group(1)
                p_ind_clms = p_clms.split("\n\n")
                p_clean_ind_clms = []
                for p_ind_clm in p_ind_clms:
                    if "(canceled)" in p_ind_clm:
                        continue
                    else:
                        p_clean_ind_clms.append(self.clean_text(p_ind_clm))
                p_text_input = "[IPC]" + p_ipc + "[TTL]" + p_ttl
                for i in range(len(p_clean_ind_clms)):
                    p_text_input += "[CLMS]" + p_clean_ind_clms[i]
                encoded_p = self.tokenizer(p_text_input, return_tensors="pt", max_length=self.chunk_size, padding="max_length", truncation=True)

                                n_ttl = re.search("<TTL>([\s\S]*?)<IPC>", n).group(1)
                n_ttl = n_ttl.lower()
                n_ipc = re.search("<IPC>([\s\S]*?)<ABST>", n).group(1)
                n_ipc = n_ipc[:3]
                n_clms = re.search("<CLMS>([\s\S]*?)<DESC>", n).group(1)
                n_ind_clms = n_clms.split("\n\n")
                n_clean_ind_clms = []
                for n_ind_clm in n_ind_clms:
                    if "(canceled)" in n_ind_clm:
                        continue
                    else:
                        n_clean_ind_clms.append(self.clean_text(n_ind_clm))
                n_text_input = "[IPC]" + n_ipc + "[TTL]" + n_ttl
                for i in range(len(n_clean_ind_clms)):
                    n_text_input += "[CLMS]" + n_clean_ind_clms[i]
                encoded_n = self.tokenizer(n_text_input, return_tensors="pt", max_length=self.chunk_size, padding="max_length", truncation=True)

                input_ids.append(encoded_q["input_ids"])
                attn_masks.append(encoded_q["attention_mask"])
                labels.append(ids*2)

                input_ids.append(encoded_p["input_ids"])
                attn_masks.append(encoded_p["attention_mask"])
                labels.append(ids*2)

                input_ids.append(encoded_n["input_ids"])
                attn_masks.append(encoded_n["attention_mask"])
                labels.append(ids*2 + 1)
                ids += 1
            except:
                continue
        input_ids = torch.stack(input_ids, dim=0).squeeze(dim=1)
        attn_masks = torch.stack(attn_masks, dim=0).squeeze(dim=1)
        labels = torch.tensor(labels, dtype=int)
        return input_ids, attn_masks, labels

class NeuralRanker(pl.LightningModule):
    def __init__(self, hparams=dict(), plm="tanapatentlm/patentdeberta_large_spec_128_pwi", loss_type="ContrastiveLoss", use_miner=True):
        super(NeuralRanker, self).__init__()
        self.hparams.update(hparams)
        self.save_hyperparameters(ignore="hparams")
        self.tokenizer = AutoTokenizer.from_pretrained(plm)
        self.config = AutoConfig.from_pretrained(plm)
        self.loss_type = loss_type
        self.use_miner = use_miner
        self.net = AutoModel.from_pretrained(plm)
        if self.loss_type == "ContrastiveLoss":
            self.metric = losses.ContrastiveLoss()
            # # change distance metric to cosine similarity
            # self.metric = losses.ContrastiveLoss(pos_margin=1, neg_margin=0, distance=CosineSimilarity())
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

    # for inference
    def predict_step(self, batch, batch_idx: int, dataloader_idx: int=0):
        q_input_ids, q_attn_masks = batch["q"]["input_ids"], batch["q"]["attention_mask"]
        q_emb = self(q_input_ids, q_attn_masks)
        return q_emb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", "-s", type=str, default="deberta_large_default.yaml", help="Experiment settings")
    args = parser.parse_args(args=[])
    hparams = addict.Addict(dict(load_hparams_from_yaml(args.setting)))


    train_set = TripletData("./FGH_spec_ind_claim_triplet_v1.4.1s/train_triplet.csv")
    valid_set = TripletData("./FGH_spec_ind_claim_triplet_v1.4.1s/valid_triplet.csv")
    collate = custom_collate(is_test=False)

    train_dataloader = DataLoader(train_set, batch_size=hparams.batch_size, collate_fn=collate, shuffle=True)
    valid_dataloader = DataLoader(valid_set, batch_size=hparams.batch_size, collate_fn=collate, shuffle=False)

    model = NeuralRanker(hparams,
                         loss_type = "MultiSimilarityLoss")

    ckpt_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        dirpath="./checkpoint/v1.4.1s_experiments/DeBERTa_Large/MultiSimilarityLoss/",
        filename="epoch_end_checkpoints-{epoch:02d}-{val_loss:.8f}",
        save_top_k=3,
        mode="min",
        save_last=True  # When ``True``, saves an exact copy of the checkpoint to a file `last.ckpt` whenever a checkpoint file gets saved.
    )

    ckpt_callback = pl.callbacks.ModelCheckpoint(
        monitor="train_loss",
        dirpath="./checkpoint/v1.4.1s_experiments/DeBERTa_Large/MultiSimilarityLoss/",
        every_n_train_steps = 1000,
        filename="intermediate_checkpoints-{epoch:02d}-{step:02d}-{train_loss:.8f}",
        save_top_k=5,
        mode="min",
        save_last=True
    )

    SWA = pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2)

    device_cnt = torch.cuda.device_count()
    print("device count = {}".format(device_cnt))

    trainer = pl.Trainer(gpus=device_cnt,
                         max_epochs=hparams.epochs,
                         strategy="ddp" if device_cnt > 1 else None,
                         callbacks=[ckpt_callback, ckpt_callback_steps],
                         gradient_clip_val=1.0,
                         accumulate_grad_batches=10,
                         num_sanity_val_steps=20)

    print("Start training model!")
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
