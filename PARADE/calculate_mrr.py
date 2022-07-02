import os 
import pickle 
from pathlib import Path 
import torch 
from torch.utils.data import DataLoader 
import pytorch_lightning as pl 
from pytorch_lightning.core.saving import load_hparams_from_yaml 
import addict 
import numpy as np 
import pandas as pd 
from scipy.spatial import distance 
import tqdm 
from tqdm.contrib.concurrent import process_map 
from functools import partial 
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

    def chunk_tokens(self, tokens, overlap=12, chunk_size=128):
        total, partial = [], []
        if len(tokens) // (chunk_size - overlap) > 0:
            n = len(tokens) // (chunk_size - overlap)
        else:
            n = 1
        for w in range(n):
            if w == 0:
                partial = tokens[:chunk_size]
            else:
                partial = tokens[w*(chunk_size - overlap):w*(chunk_size-overlap)+chunk_size]
            total.append(partial)
        return total

    def get_chunked(self, t):
        input_ids = self.tokenizer(t)['input_ids']
        attention_mask = self.tokenizer(t)['attention_mask']
        chunks = self.chunk_tokens(input_ids)
        chunk_attention_mask = self.chunk_tokens(attention_mask)
        if len(chunks) > self.seq_len:
            chunks = chunks[:self.seq_len]
            chunk_attention_mask = chunk_attention_mask[:self.seq_len]
        else:
            while len(chunks) < self.seq_len:
                chunks.append([])
                chunk_attention_mask.append([])

        for i in range(len(chunks)):
            while len(chunks[i]) < self.window_size:
                chunks[i].append(0) # add padding tokens
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
        # print(x)

        # print(self.pe)

        self.pe = self.pe.to(self.device)

        # print(self.pe)

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
        x = torch.permute(x, (1,0,2)) # shape: [1, 20, 768]

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
        self.log('val_loss', 1) 
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
        self.log('val_loss', avg_loss) 
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



# code for calculating MRR 
with open('../storage/deberta_embeddings/epoch0_embedding.pkl','rb') as f:
    embeddings = pickle.load(f)
    q_v = embeddings['query']
    candidate = embeddings['candidate']

def get_rank(inp, candidate):
    i, q = inp
    distances = distance.cdist([q], candidate.copy(), "cosine")[0]
    rank = np.argsort(distances)
    # return (p1 index, positive rank) tuple
    return rank[0], np.where(rank==i)[0][0] + 1
    

ranks = process_map(partial(get_rank, candidate=candidate),
                            enumerate(q_v),
                            total=len(q_v),
                            max_workers=32)

p1, rank = zip(*ranks)

result = pd.DataFrame()

result['p1'] = p1
result['rank'] = rank 
result['r_rank'] = 1 / result['rank'] 
total_count = result.count()['rank']
for i, r in enumerate([1,3,5,10,20,30,50,100]):
    subset = result.apply(lambda x : x['r_rank'] if int(x['rank']) <= r else 0, axis=1) 
    mrr = subset.sum() 
    mrr_count = subset.astype(bool).sum()
    print(f'MRR@{r}:', mrr / total_count, '/ count:', mrr_count)

print('average rank {}'.format(result['rank'].sum() / total_count)) 

result.to_csv("../storage/evaluation_results/deberta_specs_parade_epoch_0_results.csv", index=False)
