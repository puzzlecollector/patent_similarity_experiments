import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel

class PolyEncoder(BertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.bert = kwargs["bert"]
        self.poly_m = kwargs["poly_m"]
        self.poly_code_embeddings = nn.Embedding(self.poly_m, config.hidden_size)
        torch.nn.init.normal_(self.poly_code_embeddings.weight, config.hidden_size ** -0.5)

    def dot_attention(self, q, k, v):
        attn_weights = torch.matmul(q, k.transpose(2,1))
        attn_weights = F.softmax(attn_weights, -1)
        output = torch.matmul(attn_weights, v)
        return output

    def forward(self, context_input_ids, context_input_masks, responses_input_ids, responses_input_masks, labels=None):
        batch_size, res_cnt, seq_length = responses_input_ids.shape
        ctx_out = self.bert(context_input_ids, contenxt_input_masks)[0] # [bs, length, dim]
        poly_code_ids = torch.arange(self.poly_m, dtype=torch.long).to(context_input_ids.device)
        poly_code_ids = poly_code_ids.unsqueeze(0).expand(batch_size, self.poly_m)
        poly_codes = self.poly_code_embeddings(poly_code_ids) # [bs, poly_m, dim]
        embs = self.dot_attention(poly_codes, cxt_out, ctx_out) # [bs, poly_m, dim]

        responses_input_ids = responses_input_ids.view(-1, seq_length)
        responses_input_masks = responses_input_masks.view(-1, seq_length)
        cand_emb = self.bert(responses_input_ids, responses_input_masks)[0][:,0,:] # [bs, dim]
        cand_emb = cand_emb.view(batch_size, res_cnt, -1) # [bs, res_cnt, dim]

        ctx_emb = self.dot_attention(cand_emb, embs, embs) # [bs, res_cnt, dim]
        dot_product = (ctx_emb * cand_emb).sum(-1)
        return dot_product
