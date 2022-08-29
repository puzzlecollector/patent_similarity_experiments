"""
PatentSBERTa
pytorch-lightning module

Author: YongWook Ha
"""
import torch
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModel, AutoConfig

class PatentSBERTa(pl.LightningModule):
    def __init__(self, hparams=dict(), is_train=True):
        """initialize

        Args:
            hparams (_type_, optional): _description_. Defaults to dict().
                if 'model_pretrained' in the haparams, load it.
        """
        super(PatentSBERTa, self).__init__()
        self.hparams.update(hparams)
        self.save_hyperparameters(ignore='hparams')

        self.metric = torch.nn.TripletMarginLoss()

        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams['from_pretrained_tok'])
        if is_train:
            self.net = AutoModel.from_pretrained(self.hparams['from_pretrained_model'])
        else:
            print('### loading from checkpoint_PatentSBERTa')
            config = AutoConfig.from_pretrained(self.hparams['from_pretrained_model'])
            self.net = AutoModel.from_config(config)
            state_dict = torch.load(hparams['checkpoint'], map_location=self.device)['state_dict']
            net = {}
            for k, v in state_dict.items():
                if k.startswith('net.'):
                    net[k[4:]] = v
            print(self.net.load_state_dict(net))
            self.net.eval()

        if 'additional_special_tokens' in self.hparams and self.hparams['additional_special_tokens']:
            additional_special_tokens = self.hparams['additional_special_tokens']
            self.tokenizer.add_special_tokens({'additional_special_tokens': additional_special_tokens})
            self.net.resize_token_embeddings(len(self.tokenizer))

    def forward(self, input):
        """
        input:
            [B, ]
        return:
            [B, ]
        """
        encoded_input = self.tokenizer(list(input), return_tensors='pt', padding='max_length', truncation=True)
        model_output = self.net(**encoded_input.to(self.device))
        return model_output['last_hidden_state'][:, 0, :]  # cls-pooling  [N, W, E]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=float(self.hparams.lr),
                          weight_decay=float(self.hparams.weight_decay), eps=1e-8)
        return [optimizer]

    def training_step(self, batch, batch_nb):
        q, pos, neg = zip(*batch)

        q_emb = self(q)
        pos_emb = self(pos)
        neg_emb = self(neg)
        loss = self.cal_loss(q_emb, pos_emb, neg_emb)

        self.log('train_loss', loss, batch_size=len(batch))
        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        q, pos, neg = zip(*batch)

        q_emb = self(q)
        pos_emb = self(pos)
        neg_emb = self(neg)
        loss = self.cal_loss(q_emb, pos_emb, neg_emb)

        self.log('val_loss', loss, batch_size=len(batch))

        return {'val_loss':loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        print(f"\nEpoch {self.current_epoch} | avg_loss:{avg_loss}\n")

    def predict_step(self, query, batch_idx: int, dataloader_idx: int = 0):      
        return self(query)

    def cal_loss(self, query, positive, negative):
        """
        Define how to calculate loss

        logits:
            [B, ]
        targets:
            [B, ]
        """
        loss = self.metric(query, positive, negative)

        return loss

class Doc_ranker(PatentSBERTa):
    def __init__(self, hparams=dict(), is_train=True):
        """initialize

        Args:
            hparams (_type_, optional): _description_. Defaults to dict().
                if 'model_pretrained' in the haparams, load it.
        """
        super(Doc_ranker, self).__init__(hparams, is_train)
        self.hparams.update(hparams)
        self.save_hyperparameters(ignore='hparams')

        self.metric = torch.nn.TripletMarginLoss()

        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams['from_pretrained_tok'])
        if is_train:
            self.model = AutoModel.from_pretrained(self.hparams['from_pretrained_model'])
        else:
            print('### loading from checkpoint')
            config = AutoConfig.from_pretrained(self.hparams['from_pretrained_model'])
            self.model = AutoModel.from_config(config)
            state_dict = torch.load(hparams['checkpoint'], map_location=self.device)['state_dict']
            new_weights = self.model.state_dict()
            old_weights = list(state_dict.items())
            i = 0
            for k, _ in new_weights.items():
                new_weights[k] = old_weights[i][1]
                i += 1
            print(self.model.load_state_dict(new_weights))
            self.net = self.model

            del self.model
            try:
                print(self.model)
            except:
                print('self.model deleted')
            self.net.eval()

        if 'additional_special_tokens' in self.hparams and self.hparams['additional_special_tokens']:
            additional_special_tokens = self.hparams['additional_special_tokens']
            self.tokenizer.add_special_tokens({'additional_special_tokens': additional_special_tokens})
            self.net.resize_token_embeddings(len(self.tokenizer))

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, inputs):
        model_output = self.net(**inputs.to(self.device))
        model_output = self.mean_pooling(model_output, inputs['attention_mask'])
        return model_output

    def predict_step(self, query, batch_idx: int, dataloader_idx: int = 0):
        q = query.pop('q')
        q_emb = self(q)
        if query.get('p') is not None:
            p = query.pop('p')
            p_emb = self(p)
            n = query.pop('n')
            n_emb = self(n)
            return q_emb, p_emb, n_emb
        return q_emb
