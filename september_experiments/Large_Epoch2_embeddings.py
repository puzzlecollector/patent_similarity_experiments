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


torch.cuda.empty_cache()

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
    def __init__(self, plm="tanapatentlm/patentdeberta_large_spec_128_pwi", is_debug=False):
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

data_path = Path("../storage/FGH_spec_ind_claim_triplet_v1.4.1s/")
model_pt_path = Path("val_loss=0.19774973.ckpt")
emb_dim = 1024
output_dir = Path("../storage/DeBERTa_Large_embeddings_epochs2")
debug = False

### define dataloader ###
dataset = TripletData(data_path, debug)
collate = custom_collate(is_debug=False)
dataloader = DataLoader(dataset, batch_size=300, collate_fn=collate, shuffle=False)

### load model ###
print("loading model...")
parser = argparse.ArgumentParser()
parser.add_argument("--setting", "-s", type=str, default="deberta_large_default.yaml", help="Experiment settings")
args = parser.parse_args(args=[])
hparams = addict.Addict(dict(load_hparams_from_yaml(args.setting)))
model = NeuralRanker(hparams,
                     loss_type = "MultiSimilarityLoss")

device = torch.device("cuda")
checkpoint = torch.load(model_pt_path, map_location=device)
loaded_dict = checkpoint["state_dict"]
model = NeuralRanker(hparams,
                     loss_type = "MultiSimilarityLoss")
model.load_state_dict(loaded_dict)
model.eval()
model.freeze()

### inference callback ###
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


prediction_callback = CustomWriter(output_dir, write_interval="batch")

device_cnt = torch.cuda.device_count()

print("device count = {}".format(device_cnt))

trainer = pl.Trainer(accelerator="gpu",
                     devices=device_cnt,
                     strategy="ddp" if device_cnt > 1 else None,
                     callbacks=[prediction_callback],
                     max_epochs=1)


trainer.predict(model, dataloaders=dataloader, return_predictions=False)