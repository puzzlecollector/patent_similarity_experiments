# fixed error of not including CLS tokens to chunks 
# gradient clipping to avoid gradient explosion-like behavior 
from xml.dom import ValidationErr
from torch.utils.data import Dataset, TensorDataset, DataLoader
from pathlib import Path
import torch
import pytorch_lightning as pl
from transformers import *
import torch
import argparse
import addict
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.core.saving import load_hparams_from_yaml, update_hparams
import os
from tqdm import tqdm
from torch import nn, Tensor
import math
import time
import datetime
import pandas as pd 
import numpy as np 

os.environ["TOKENIZERS_PARALLELISM"] = "true"

class TripletData(Dataset):
    def __init__(self, path):
        super(TripletData, self).__init__()
        self.data = [txt for txt in Path(path).glob('*.txt')]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class custom_collate(object):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("tanapatentlm/patentdeberta_base_spec_1024_pwi")
        self.seq_len = 10
        self.window_size = 128

    def chunk_tokens(self, tokens, start_token_id, end_token_id, overlap=12, chunk_size=128):
        chunk_size = chunk_size - self.tokenizer.num_special_tokens_to_add()
        total, partial = [], []
        if len(tokens) / (chunk_size - overlap) > 0:
            n = math.ceil(len(tokens) / (chunk_size - overlap))
        else:
            n= 1
        for w in range(n):
            if w == 0:
                partial = tokens[:chunk_size]
            else:
                partial = tokens[w * (chunk_size - overlap):w * (chunk_size - overlap) + chunk_size]
            partial = [start_token_id] + partial + [end_token_id]
            total.append(partial)
        return total
    
    def get_chunked(self, t):
        tokenizer_outputs = self.tokenizer(t, add_special_tokens=False)
        input_ids = tokenizer_outputs['input_ids']
        attention_mask = tokenizer_outputs['attention_mask']
        chunks = self.chunk_tokens(input_ids, self.tokenizer.cls_token_id, self.tokenizer.sep_token_id)
        chunk_attention_mask = self.chunk_tokens(attention_mask, 1, 1)
        if len(chunks) > self.seq_len: 
            chunks = chunks[:self.seq_len]
            chunk_attention_mask = chunk_attention_mask[:self.seq_len]
        else:
            while len(chunks) < self.seq_len:
                chunks.append([])
                chunk_attention_mask.append([])

        for i in range(len(chunks)):
            while len(chunks[i]) < self.window_size:
                chunks[i].append(0)  # add padding tokens
                chunk_attention_mask[i].append(0)
            chunks[i] = torch.tensor(chunks[i], dtype=int)
            chunk_attention_mask[i] = torch.tensor(chunk_attention_mask[i], dtype=int)
        return chunks, chunk_attention_mask

    def tensor_format(self, x):
        out_tensor = torch.Tensor(self.seq_len, self.window_size)
        torch.cat(x, out = out_tensor)
        out_tensor = torch.reshape(out_tensor, (self.seq_len, self.window_size))
        out_tensor = out_tensor.int()
        return out_tensor

    def __call__(self, batch):
        """
        ret (anchor input ids, anchor attention masks,
             positive input ids, positive attention masks,
             negative input ids, negative attention masks)
        """
        ret = []
        for txt_file in batch:
            with txt_file.open('r', encoding='utf8') as f:
                data = f.read()
            # triplet: (query, positive, negative)
            triplet = data.split('\n\n\n')
            if len(triplet) != 3:
                print(f'[Warning] len(triplet) != 3, skipped this data [txt_file]')
            q, p, n = triplet[0], triplet[1], triplet[2]
            q_chunk_input_ids, q_chunk_attention_masks = self.get_chunked(q)
            p_chunk_input_ids, p_chunk_attention_masks = self.get_chunked(p)
            n_chunk_input_ids, n_chunk_attention_masks = self.get_chunked(n)


            q_chunk_input_ids_torch = self.tensor_format(q_chunk_input_ids)
            q_chunk_attention_masks_torch = self.tensor_format(q_chunk_attention_masks)

            p_chunk_input_ids_torch = self.tensor_format(p_chunk_input_ids)
            p_chunk_attention_masks_torch = self.tensor_format(p_chunk_attention_masks)

            n_chunk_input_ids_torch = self.tensor_format(n_chunk_input_ids)
            n_chunk_attention_masks_torch = self.tensor_format(n_chunk_attention_masks)


            dataset = [q_chunk_input_ids_torch, q_chunk_attention_masks_torch,
                       p_chunk_input_ids_torch, p_chunk_attention_masks_torch,
                       n_chunk_input_ids_torch, n_chunk_attention_masks_torch]
            ret.append(dataset)
        return ret

'''
Attentive pooling
'''
class AttentivePooling(torch.nn.Module):
    def __init__(self, input_dim):
        super(AttentivePooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)

    def forward(self, x):
        """
        input:
            batch_rep : size (N,T,H), N: batch size, T: sequence length, H: hidden dimension
        attention weight:
            att_w : size (N,T,1)
        return:
            size (N, H)
        """
        softmax = nn.functional.softmax
        att_w = softmax(self.W(x).squeeze(-1)).unsqueeze(-1)
        x = torch.sum(x * att_w, dim=1)
        return x

'''
Positional Encoding
'''
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model:int, dropout:float=0.1, max_len:int=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.pe = torch.zeros(max_len, 1, d_model)
        self.pe[:, 0, 0::2] = torch.sin(position * div_term)
        self.pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.device = torch.device('cuda')


    def forward(self, x:Tensor) -> Tensor:
        '''
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        '''
        self.pe = self.pe.to(self.device)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


'''
Simplified Hi-Transformer architecture (closer to PARADE)
M = 10, K = 128 tokens
using only claims data for fine-tuning
'''
class Hi_DeBERTa_Ranker(pl.LightningModule):
    def __init__(self, train_dataloader_length, hparams=dict()):
        super(Hi_DeBERTa_Ranker, self).__init__()
        self.hparams.update(hparams)
        self.save_hyperparameters(ignore='hparams')
        self.chunk_LM = AutoModel.from_pretrained("tanapatentlm/patentdeberta_base_spec_1024_pwi")

        print("loading from checkpoint...")
        ckpt_path = "../storage/checkpoints/deberta_base_spec_1024_pwi/checkpoints-epoch=08-val_loss=0.06.ckpt"
        checkpoint = torch.load(ckpt_path)
        new_weights = self.chunk_LM.state_dict()
        old_weights = list(checkpoint['state_dict'].items())
        i=0
        for k, _ in new_weights.items():
            new_weights[k] = old_weights[i][1]
            i += 1
        self.chunk_LM.load_state_dict(new_weights)

        self.pos_encoder = PositionalEncoding(d_model=768)
        encoder_layers = nn.TransformerEncoderLayer(d_model=768, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=6)
        self.attentive_pooling = AttentivePooling(768)
        self.metric = torch.nn.TripletMarginLoss()
        self.seq_len = 10
        self.window_size = 128
        self.train_dataloader_length = train_dataloader_length

    def forward(self, input_ids, attention_mask):
        #x = self.chunk_LM(input_ids=input_ids[:,0,:],attention_mask=attention_mask[:,0,:])

        x = self.chunk_LM(input_ids=input_ids[:][0], attention_mask=attention_mask[:][0])
        x = x[0][:,0] # CLS pooling
        x = torch.unsqueeze(x,dim=1)
        x = torch.permute(x, (1,0,2)) # shape: [1, 10, 768]

        for i in range(1, len(input_ids[:])):
            output_vec = self.chunk_LM(input_ids=input_ids[:][i], attention_mask=attention_mask[:][i])
            output_vec = output_vec[0][:,0]
            output_vec = torch.unsqueeze(output_vec,dim=1)
            output_vec = torch.permute(output_vec, (1,0,2))
            x = torch.cat((x, output_vec), dim=0) # concat across dimension 0

        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.attentive_pooling(x)
        return x

    def calc_loss(self, q, p, n):
        loss = self.metric(q, p, n)
        return loss

    def training_step(self, batch, batch_nb):
        q_input_ids, q_attn_masks, p_input_ids, p_attn_masks, n_input_ids, n_attn_masks = zip(*batch)

        q_emb = self(q_input_ids, q_attn_masks)
        p_emb = self(p_input_ids, p_attn_masks)
        n_emb = self(n_input_ids, n_attn_masks)
        loss = self.calc_loss(q_emb, p_emb, n_emb)
        self.log('train_loss', loss, batch_size=len(batch))
        return {'loss':loss}

    def validation_step(self, batch, batch_nb):
        q_input_ids, q_attn_masks, p_input_ids, p_attn_masks, n_input_ids, n_attn_masks = zip(*batch)

        q_emb = self(q_input_ids, q_attn_masks)
        p_emb = self(p_input_ids, p_attn_masks)
        n_emb = self(n_input_ids, n_attn_masks)
        loss = self.calc_loss(q_emb, p_emb, n_emb)
        self.log('val_loss', loss, batch_size=len(batch))
        return {'val_loss':loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        print(f"\nEpoch {self.current_epoch} | avg_val_loss:{avg_loss}\n")
        
    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        q_input_ids, q_attn_masks, p_input_ids, p_attn_masks, n_input_ids, n_attn_masks = zip(*batch)
        q_emb = self(q_input_ids, q_attn_masks)
        p_emb = self(p_input_ids, p_attn_masks)
        n_emb = self(n_input_ids, n_attn_masks)
        return q_emb, p_emb, n_emb

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=float(self.hparams["lr"]), eps=1e-8)
        train_steps = self.train_dataloader_length * self.hparams['epochs']
        lr_scheduler = get_linear_schedule_with_warmup(optimizer,
                                                       num_warmup_steps = int(0.1 * train_steps),
                                                       num_training_steps = train_steps)
        return [optimizer], [{"scheduler": lr_scheduler, "interval":"step"}]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", "-s", type=str, default="../storage/patent_experiments/default.yaml",
                        help="Experiment settings")
    parser.add_argument("--num_workers", "-nw", type=int, default=4, help="Number of workers for dataloader")
    parser.add_argument("--resume_train", "-rt", type=str, default="", help="Resume train from certain checkpoint")
    args = parser.parse_args()
    print(vars(args))

    inter = load_hparams_from_yaml(args.setting)
    inter = dict(inter)
    hparams = addict.Addict(inter)
    update_hparams(hparams, vars(args))

    train_set = TripletData("../storage/patent_experiments/FGH_claim_triplet_v0.1s/train")
    val_set = TripletData("../storage/patent_experiments/FGH_claim_triplet_v0.1s/valid")
    collate = custom_collate()

    train_dataloader = DataLoader(train_set, batch_size = hparams.batch_size, num_workers = hparams.num_workers, collate_fn = collate, shuffle = True)
    valid_dataloader = DataLoader(val_set, batch_size = hparams.batch_size, num_workers = hparams.num_workers, collate_fn = collate, shuffle = False)

    model = Hi_DeBERTa_Ranker(int(len(train_dataloader)), hparams)

    if hparams.resume_train:
        model = model.load_from_checkpoint(hparams.resume_train)

    logger = TensorBoardLogger("tb_logs", name="model", version=hparams.version,
                               default_hp_metric=False)

    ckpt_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        dirpath="../storage/checkpoints",
        filename="[data:claim]_[model:deberta_base_spec_1024]_[method:cls]-{epoch:02d}-{train_loss:.8f}-{val_loss:.8f}",
        save_top_k=3,
        mode="min",
    )

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval='step')
    device_cnt = torch.cuda.device_count()
    trainer = pl.Trainer(gpus=device_cnt, 
                         max_epochs=hparams.epochs,
                         gradient_clip_val = 1.0, 
                         accumulate_grad_batches = 10, 
                         logger=logger, 
                         num_sanity_val_steps=1,
                         strategy="ddp" if device_cnt > 1 else None,
                         callbacks=[ckpt_callback, lr_callback],
                         resume_from_checkpoint=hparams.resume_train if hparams.resume_train else None)

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
