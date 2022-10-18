import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time
import datetime
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.core.saving import load_hparams_from_yaml, update_hparams
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoConfig, AutoModel, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, AutoModelForSequenceClassification
import addict
import argparse
from tqdm.auto import tqdm
import random
from sklearn.utils.class_weight import compute_class_weight

train_data = torch.load("trainset_over_10000.pt") 
valid_data = torch.load("validset_over_10000.pt") 

train_labels = train_data[1]
valid_labels = valid_data[1]

ipc_to_id = {}
freq_dict = {} # store frequency of each class 

id_num = 0

for i in tqdm(range(len(train_labels))):
    if train_labels[i][0] not in ipc_to_id.keys():
        ipc_to_id[train_labels[i][0]] = id_num
        id_num += 1 
    if train_labels[i][0] not in freq_dict.keys(): 
        freq_dict[train_labels[i][0]] = 1 
    else: 
        freq_dict[train_labels[i][0]] += 1 

for i in tqdm(range(len(valid_labels))):
    if valid_labels[i][0] not in ipc_to_id.keys():
        ipc_to_id[valid_labels[i][0]] = id_num
        id_num += 1
    if valid_labels[i][0] not in freq_dict.keys(): 
        freq_dict[valid_labels[i][0]] = 1 
    else: 
        freq_dict[valid_labels[i][0]] += 1 

train_label_ids = []
for i in tqdm(range(len(train_labels))):
    train_label_ids.append(ipc_to_id[train_labels[i][0]]) 
    
valid_label_ids = [] 
for i in tqdm(range(len(valid_labels))): 
    valid_label_ids.append(ipc_to_id[valid_labels[i][0]])

class_weights = compute_class_weight(class_weight = "balanced", classes = np.unique(train_label_ids), y=np.array(train_label_ids))
class_weights = torch.tensor(class_weights, dtype=torch.float) 

print(f"class weights: {class_weights}") 
print(f"class ids: {ipc_to_id}") 
print(f"class frequencies: {freq_dict}") 


class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.device = torch.device("cuda")
        self.alpha = self.alpha.to(self.device) # move to gpu
        self.gamma = gamma
    def forward(self, inputs, targets):
        CE_loss = nn.CrossEntropyLoss()(inputs, targets)
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-CE_loss)
        F_loss = at * (1-pt)**self.gamma * CE_loss
        return F_loss.mean()

class IPCDataset(Dataset):
    def __init__(self, arr, ipc_to_id):
        super().__init__()
        text_paths, labels = arr
        self.data = []
        self.ipc_to_id = ipc_to_id
        for idx, text_path in enumerate(tqdm(text_paths)):
            try:
                data = []
                data.append(f"FGH_spec_ind_claim_triplet_v1.4.1s/{text_path}")
                data.append(ipc_to_id[labels[idx][0]]) 
                self.data.append(data)
            except:
                continue

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class custom_collate(object):
    def __init__(self, plm="AI-Growth-Lab/PatentSBERTa", id_dict = ipc_to_id):
        self.tokenizer = AutoTokenizer.from_pretrained(plm)
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["[IPC]", "[TTL]", "[CLMS]", "[ABST]"]})
        self.chunk_size = 512
        self.id_dict = ipc_to_id
    def clean_text(self, t):
        x = re.sub("\d+","",t)
        x = x.replace("\n", " ")
        x = x.strip()
        return x
    def __call__(self, batch):
        input_ids, attn_masks, labels = [], [], []
        for idx, (text_path, label) in enumerate(batch):
            try:
                with Path(text_path).open("r", encoding="utf8") as f:
                    text = f.read()
                ttl = re.search("<TTL>([\s\S]*?)<IPC>", text).group(1)
                ttl = ttl.lower()
                clms = re.search("<CLMS>([\s\S]*?)<DESC>", text).group(1)
                clms = clms.split("\n\n")
                clean_clms = []
                for clm in clms:
                    if "(canceled)" in clm:
                        continue
                    else:
                        clean_clms.append(self.clean_text(clm))
                text_input = "[TTL]" + ttl
                for i in range(len(clean_clms)):
                    text_input += "[CLMS]" + clean_clms[i]

                encoded_input = self.tokenizer(text_input, return_tensors="pt", max_length=self.chunk_size, padding="max_length", truncation=True)
                input_ids.append(encoded_input["input_ids"])
                attn_masks.append(encoded_input["attention_mask"])
                labels.append(label)
            except:
                continue
        input_ids = torch.stack(input_ids, dim=0).squeeze(dim=1)
        attn_masks = torch.stack(attn_masks, dim=0).squeeze(dim=1)
        labels = torch.tensor(labels)

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
    def __init__(self, num_hidden_layers, layer_start: int = 4, layer_weights=None):
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
    def __init__(self, hparams=dict(), plm="AI-Growth-Lab/PatentSBERTa", num_classes=50, class_weights=class_weights):
        super(Classifier, self).__init__()
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.hparams.update(hparams)
        self.save_hyperparameters(ignore="hparams")
        self.tokenizer = AutoTokenizer.from_pretrained(plm)
        self.config = AutoConfig.from_pretrained(plm)
        self.net = AutoModel.from_pretrained(plm, config=self.config)
        self.mean_pooling = MeanPooling()
        self.weighted_layer_pooling = WeightedLayerPooling(self.config.num_hidden_layers, 9, None)
        self.fc = nn.Linear(self.config.hidden_size, self.num_classes)
        self._init_weights(self.fc)
        self.multi_dropout = MultiSampleDropout(0.2, 8, self.fc)
        # self.metric = nn.CrossEntropyLoss()
        self.metric = WeightedFocalLoss(alpha=self.class_weights)

        if "additional_special_tokens" in self.hparams and self.hparams["additional_special_tokens"]:
            print("adding additional special tokens")
            additional_special_tokens = self.hparams["additional_special_tokens"]
            self.tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})
            self.net.resize_token_embeddings(len(self.tokenizer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mena=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


    def forward(self, input_ids, attention_mask):
        x = self.net(input_ids, attention_mask, output_hidden_states=True)
        x = self.weighted_layer_pooling(x.hidden_states)
        x = x[:,0]
        x = self.multi_dropout(x)
        return x

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
        scheduler = {"scheduler": scheduler, "interval":"step", "frequency":1}
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
        return {"val_loss":loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        print(f"\nEpoch {self.current_epoch} | avg_loss:{avg_loss}\n")

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int=0):
        sample_input_ids, sample_attn_masks, sample_labels = batch
        return (self(sample_input_ids, sample_attn_masks), sample_labels) 

      
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", "-s", type=str, default="clf_default.yaml", help="Experiment settings")
    args = parser.parse_args(args=[])
    hparams = addict.Addict(dict(load_hparams_from_yaml(args.setting)))

    train_set = IPCDataset(train_data, ipc_to_id)
    valid_set = IPCDataset(valid_data, ipc_to_id)
    collate = custom_collate()

    train_dataloader = DataLoader(train_set, batch_size=hparams.batch_size, collate_fn=collate, shuffle=True)
    valid_dataloader = DataLoader(valid_set, batch_size=hparams.batch_size, collate_fn=collate, shuffle=False)

    model = Classifier(hparams)

    ckpt_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        dirpath="./clf_test_checkpoint/",
        filename="clf_chkpt_{epoch:02d}_{val_loss:.8f}",
        save_top_k=3,
        mode="min",
        save_last=True
    )

    

    device_cnt = torch.cuda.device_count()
    trainer = pl.Trainer(gpus=device_cnt,
                         max_epochs=hparams.epochs,
                         strategy="ddp" if device_cnt > 1 else None,
                         callbacks=[ckpt_callback],
                         gradient_clip_val=1.0,
                         accumulate_grad_batches=10,
                         num_sanity_val_steps=30)

    print("start training model!")
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
