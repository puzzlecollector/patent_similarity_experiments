import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time
import datetime
from pathlib import Path

# from test_umap import DimensionReducer
import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.core.saving import load_hparams_from_yaml, update_hparams
from pytorch_lightning.loggers import TensorBoardLogger

from transformers import AutoTokenizer, AutoConfig, AutoModel, get_linear_schedule_with_warmup

import addict
import argparse
import random 

class ClassificationDataset(Dataset):
    def __init__(self, root:str, ipc_len:int, sample_num:int = 99999999):
        super().__init__()
        self.data = []
        self.ipcs = []
        self.ipc_len = ipc_len
        self.ipc_cnt = {}
        data_file_list = list(Path(root).glob("*.txt"))
        if sample_num < 99999999:
            random.shuffle(data_file_list)

        for idx, fn in enumerate(data_file_list):
            ipc = self.check_ipc(fn)
            if self.ipc_cnt.get(ipc, 0) < sample_num:
                if self.ipc_cnt.get(ipc) is None:
                    self.ipc_cnt[ipc] = 1
                else:
                    self.ipc_cnt[ipc] += 1
                self.data.append(fn)
                self.ipcs.append(ipc)
            if list(v for v in self.ipc_cnt.values()) == [sample_num] * len(self.ipc_cnt.keys()):
                break
        print(self.ipc_cnt)
        print(idx)
        print(sum(list(v for v in self.ipc_cnt.values())))

    def __getitem__(self, index):
        return self.data[index], self.ipcs[index]
    
    def __len__(self):
        return len(self.data)    
    
    def check_ipc(self, doc_path):
        with(doc_path).open('r', encoding='utf8') as f:
            d = f.read()
        ipc = re.search("<IPC>([\s\S]*?)<ABST>", d).group(1)
        ipc = ipc[:self.ipc_len].upper()
        return ipc

class DropedClassification(ClassificationDataset):
    def __init__(self, root:str, data_path: str):
        root = Path(root)
        data_files, self.ipcs = torch.load(data_path)
        self.data = [root/file for file in data_files]
        # self.ipcs = self.ipcs[:1000]
        # self.data = self.data[:1000]

    def __getitem__(self, index):
        return super().__getitem__(index)
    
    def __len__(self):
        return super().__len__()

class abs_collate(object):
    def __init__(self,
                 max_depth=3,
                 chunk_size=512,
                 plm="tanapatentlm/patentdeberta_base_spec_1024_pwi",
                 label_list:set=None):
        self.tokenizer = AutoTokenizer.from_pretrained(plm)
        # regular token -> special token
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["[IPC]", "[TTL]", "[CLMS]", "[ABST]"]})
        self.chunk_size = chunk_size
        self.max_depth = max_depth
        if label_list is not None:
            self.setting_label_dict(label_list)

    def clean_text(self, t):
        x = re.sub("\d+","",t)
        x = x.replace("\n"," ")
        x = x.strip()
        return x

    def setting_label_dict(self, label_list:list):
        """
        from ipc_label_list(list), make self.ipc2label (dict) and self.label2ipc (dict) for classification
        input:
            - label_list (list) : list of total ipc label
        return: None
        """
        raise NotImplementedError 
    
    def process_document(self, document_file:Path):
        """
        from document file path, read file, process & tokenize the document for training  
        inputs:
            - document_file (Path)
        return:
            - encoded_document (Tensor, Dict, Any)
        """
        raise NotImplementedError 
        
    def process_label(self, ipc:str):
        """
        from ipc , proceess ipc labeling for training  
        inputs:
            - ipc(str)
        return:
            - label(Tensor)
        """
        raise NotImplementedError 

    def postprocess_batch(self, **batch):
        """
        post processing .. stack tensors .. etc        
        """
        raise NotImplementedError 
    
    def __call__(self, batch):
        input_ids, attn_masks, labels = [], [], [] # for training and validation
        for idx, sample in enumerate(batch):
            fn, ipc = sample 
            encoded_d = self.process_document(fn)
            input_ids.append(encoded_d["input_ids"])
            attn_masks.append(encoded_d["attention_mask"])
            label = self.process_label(ipc)
            labels.append(label)
        input_ids, attn_masks, labels = self.postprocess_batch(input_ids, attn_masks, labels)
        return input_ids, attn_masks, labels

class hierarichical_classification_collate(abs_collate):    
    def setting_label_dict(self, ipc_list:set):
        dep1, dep2, dep3 = set(), set(), set()
        for ipc in ipc_list:
            dep1.add(ipc[:1])
            if self.max_depth >= 2:
                dep2.add(ipc[1:3])
            if self.max_depth >= 3: 
                dep3.add(ipc[3:4])
        self.ipc2label = {'dep1':{ipc:i for i,ipc in enumerate(dep1)}}
        self.label2ipc = {'dep1':{i:ipc for i,ipc in enumerate(dep1)}}
        if self.max_depth >= 2:
            self.ipc2label['dep2'] = {ipc:i for i,ipc in enumerate(dep2)}
            self.label2ipc['dep2'] = {i:ipc for i,ipc in enumerate(dep2)}
        if self.max_depth >= 3: 
            self.ipc2label['dep3'] = {ipc:i for i,ipc in enumerate(dep3)}
            self.label2ipc['dep3'] = {i:ipc for i,ipc in enumerate(dep3)}

    def process_document(self, document_file):
        with Path(document_file).open("r", encoding="utf8") as f:
            d = f.read()
        ttl = re.search("<TTL>([\s\S]*?)<IPC>", d).group(1)
        ttl = ttl.lower()
        clms = re.search("<CLMS>([\s\S]*?)<DESC>", d).group(1)
        ind_clms = clms.split("\n\n")
        clean_ind_clms = []
        for ind_clm in ind_clms:
            if "(canceled)" in ind_clm:
                continue
            else:
                clean_ind_clms.append(self.clean_text(ind_clm))
        text_input = "[IPC]" * self.max_depth +  "[TTL]" + ttl  # <ipc> * max_depth
        for i in range(len(clean_ind_clms)):
            text_input += "[CLMS]" + clean_ind_clms[i]
        encoded_d = self.tokenizer(text_input, return_tensors="pt", max_length=self.chunk_size, padding="max_length", truncation=True)
        return encoded_d

    def process_label(self, label):
        ipc = [label[:1], label[1:3], label[3:4]]
        label = list(self.ipc2label[d][c] for d,c in zip(['dep1', 'dep2', 'dep3'], ipc))
        return label
    
    def postprocess_batch(self, input_ids, attn_masks, labels):
        input_ids = torch.stack(input_ids, dim=0).squeeze(dim=1)
        attn_masks = torch.stack(attn_masks, dim=0).squeeze(dim=1)
        labels = torch.tensor(labels, dtype=int)
        return input_ids, attn_masks, labels

##############################################
class luke_collate(abs_collate):
    def make_label_dict(self, label_list):
        raise NotImplementedError 
    
    def process_document(self, document_file):
        raise NotImplementedError 
        
    def process_label(self, label):
        raise NotImplementedError 

    def postprocess_batch(self, **batch):
        raise NotImplementedError 
##############################################

class LitClassifier(pl.LightningModule):
    def __init__(self,
                 hparams=dict(),
                 plm="tanapatentlm/patentdeberta_base_spec_1024_pwi",
                 is_train=True,
                 label_dict:dict = None,
                 loss_fn=F.cross_entropy, 
                 use_miner=True):
        super(LitClassifier, self).__init__()
        self.hparams.update(hparams)
        self.save_hyperparameters(ignore=["hparams", "loss_fn"])
        self.tokenizer = AutoTokenizer.from_pretrained(plm)
        self.config = AutoConfig.from_pretrained(plm)
        self.is_train = is_train
        self.loss_fn = loss_fn 
        self.use_miner = use_miner
        self.label_dict = label_dict

        self.pred_dict = {}

        if plm == "tanapatentlm/patentdeberta_base_spec_1024_pwi": 
            print("initialize PLM from previous checkpoint trained on FGH_v0.3.1")
            self.net = AutoModel.from_pretrained(plm)
            if hparams["checkpoint"] is not None:
                state_dict = torch.load(hparams["checkpoint"], map_location=self.device)
                new_weights = self.net.state_dict()
                old_weights = list(state_dict.items())
                i = 0
                for k, _ in new_weights.items():
                    new_weights[k] = old_weights[i][1]
                    i += 1
                self.net.load_state_dict(new_weights)
        else: 
            self.net = AutoModel.from_pretrained(plm)

        # classifier
        self.set_classifier()

        if self.is_train == False:
            self.net.eval()  # change to evaluation mode
        
        if "additional_special_tokens" in self.hparams and self.hparams["additional_special_tokens"]:
            print(self.hparams["additional_special_tokens"])
            additional_special_tokens = list(self.hparams["additional_special_tokens"])
            self.tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})
            self.net.resize_token_embeddings(len(self.tokenizer))
        
    def set_classifier(self):
        raise NotImplementedError

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def forward(self, input_ids, attention_mask):
        model_output = self.net(input_ids, attention_mask)
        # model_output = self.mean_pooling(model_output, attention_mask)
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

    def _convert_labels(self, labels):
        return [labels]

    def classification_step(self, embeddings, labels, is_acc=False):
        total_logits = self._calc_logits(embeddings)
        total_labels = self._convert_labels(labels)
        total_loss = 0.0
        total_acc = 0.0
        total_pred = []
        for depth, (logits, _labels) in enumerate(zip(total_logits, total_labels)):
            total_loss += self.calc_loss(logits, _labels, depth)
            if is_acc:
                acc, preds = self.calc_acc(logits, _labels, depth)
                total_acc += acc
                total_pred.append(preds)
        if is_acc:
            pred = torch.stack(total_pred, dim=-1).squeeze()
            self.save_label_and_pred(labels, pred)
            return total_loss, total_acc
        return total_loss 

    def _calc_logits(self, embeddings):
        raise NotImplementedError
    
    def calc_loss(self, logits, labels, depth):
        loss = self.loss_fn(logits, labels)
        return loss 

    def calc_acc(self, logits, labels, depth):
        pred = logits.max(1)[-1]
        acc = torch.eq(pred.view(-1), labels.view(-1)).sum().item()
        return acc, pred 
    
    def save_label_and_pred(self, labels, preds):
        assert labels.shape == preds.shape, f'label: {labels} preds:{preds}'
        if type(labels) == torch.Tensor:
            labels = labels.tolist()
        if type(preds) == torch.Tensor:
            preds = preds.tolist()

        for l, p in zip(labels, preds):
            l = self._convert_label_to_ipc(l)
            p = self._convert_label_to_ipc(p)
            self.pred_dict[self.current_epoch]['labels'].append(l)
            self.pred_dict[self.current_epoch]['preds'].append(p)

    def _convert_label_to_ipc(self, label):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        input_ids, attn_masks, labels = batch
        embeddings = self(input_ids, attn_masks)
        loss = self.classification_step(embeddings, labels)
        self.log("train_loss", loss, batch_size=len(batch))
        return {"loss": loss}

    def on_validation_start(self):
        self.pred_dict[self.current_epoch] = {'labels': [], 'preds': []}

    def validation_step(self, batch, batch_idx):
        input_ids, attn_masks, labels = batch
        embeddings = self(input_ids, attn_masks)
        loss, accuracy = self.classification_step(embeddings, labels, is_acc=True)
        self.log("val_loss", loss, batch_size=len(batch), prog_bar=True)
        self.log("val_acc", accuracy, batch_size=len(batch), prog_bar=True)
        return {"val_loss": loss, "val_acc": accuracy}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.tensor([x["val_acc"] for x in outputs], dtype=torch.float).mean()
        print(f"\nEpoch {self.current_epoch} | avg_loss:{avg_loss} | avg_acc: {avg_acc}\n")

    def test_step(self, batch, batch_idx):
        input_ids, attn_masks, labels = batch
        embeddings = self(input_ids, attn_masks)
        loss, accuracy = self.classification_step(embeddings, labels, is_acc=True)


class hierarichical_classifier(nn.Module):
    def __init__(self, hidden_size, label_num_list, depth) -> None:
        super().__init__()
        self.depth = depth
        self.classifier_1 = nn.Linear(hidden_size, label_num_list[0])
        if self.depth >= 2:
            self.classifier_2 = nn.Linear(hidden_size, label_num_list[1])
        if self.depth >= 3:
            self.classifier_3 = nn.Linear(hidden_size, label_num_list[2])

    def forward(self,embeddings, ):
        logits_list = []
        logit = self.classifier_1(embeddings[:,1])
        logits_list.append(logit)
        if self.depth >= 2:
            logit_2 = self.classifier_2(embeddings[:,2])
            logits_list.append(logit_2)
        if self.depth >= 3:
            logit_3 = self.classifier_3(embeddings[:,3])
            logits_list.append(logit_3)
        return logits_list

class jjlClassifier(LitClassifier):
    def __init__(self,
                 hparams=dict(),
                 plm="tanapatentlm/patentdeberta_base_spec_1024_pwi",
                 is_train=True,
                 label_dict:dict=None,
                 loss_fn=F.cross_entropy, 
                 depth:int=3):
        self.depth = depth
        self.label_num_list = [len(l) for l in label_dict.values()]
        super(jjlClassifier, self).__init__(
                 hparams=hparams,
                 plm=plm,
                 is_train=is_train,
                 label_dict=label_dict,
                 loss_fn=loss_fn
                 )
        
    def set_classifier(self):
        self.classifier = hierarichical_classifier(hidden_size=self.config.hidden_size, 
                                                   label_num_list=self.label_num_list,
                                                   depth=self.depth)

    def _convert_labels(self, labels):
        total_labels = []
        total_labels.append(labels[:, 0])
        if self.depth >= 2:
            total_labels.append(labels[:,1])
        if self.depth >= 3: 
            total_labels.append(labels[:, 2])
        return total_labels 

    def _calc_logits(self, embeddings):

        embeddings = embeddings.last_hidden_state
        logits_list = self.classifier(embeddings)

        return logits_list

    def calc_acc(self, logits, labels, depth):
        if depth == 0:
            d = 'large'
        elif depth == 1:
            d = 'medium'
        elif depth ==2:
            d = 'small'
        acc, pred = super().calc_acc(logits, labels, depth)
        self.log(f"val_{d}_acc", acc, prog_bar=True)
        return acc, pred

    def _convert_label_to_ipc(self, pred):
        ipc = ""
        for d, l in zip(self.label_dict.values(), pred):
            ipc += d[l]
        return ipc

##############################################
class lukeClassifier(LitClassifier):
    def set_classifier(self):
        raise NotImplementedError

    def forward(self, input_ids, attention_mask):
        model_output = self.net(input_ids, attention_mask)
        # model_output = self.mean_pooling(model_output, attention_mask)
        return model_output
    
    def _calc_logits(self, embeddings):
        raise NotImplementedError

    def _convert_label_to_ipc(self, label):
        raise NotImplementedError 
##############################################    

class CustomWriter(Callback):
    def __init__(self):
        super().__init__()
        self.curr_valid_num = 0

    def on_validation_epoch_end(self, trainer, pl_module):
        curr_epoch = trainer.current_epoch
        results = pl_module.pred_dict[curr_epoch]
        log_file = Path(trainer.log_dir, f'valid_{self.curr_valid_num}_at_{curr_epoch}_epoch.log')
        
        with open(log_file, 'w', encoding='utf8') as f:
            f.write('[GT] - [PRED]\n')
            for l,p in zip(results['labels'], results['preds']):
                f.write(f'{l} - {p} [{l==p}]\n')
        self.curr_valid_num += 1

def make_ontology(fn_list:Path):
    ontology_dict = {'4_depth':set()}
    for fn in fn_list:
        with (fn).open('r', encoding='utf8') as f:
            d = f.read()
        ipc = re.search("<IPC>([\s\S]*?)<ABST>", d).group(1)
        ipc = ipc[:4].upper()
        ontology_dict['4_depth'].add(ipc)
    return ontology_dict 

def main(drop=True, is_test=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", "-s", type=str, default="default.yaml", help="Experiment settings")
    args = parser.parse_args(args=[])
    hparams = addict.Addict(dict(load_hparams_from_yaml(args.setting)))
    ipc_len = 4
    max_depth = 3
    sample_num = 99999999
    seed = torch.Generator().manual_seed(42)
    if drop:
        train_set = DropedClassification(root=hparams.data_path, data_path=hparams.train_data_path)
        valid_set = DropedClassification(root=hparams.data_path, data_path=hparams.valid_data_path)
        
        print('Total train docs num: ', len(train_set))
        print('Total val docs num: ', len(valid_set))
        if Path('./drop_ontology.pt').exists():
            ontology_dict = torch.load('./drop_ontology.pt')
        else:
            ontology_dict = make_ontology(train_set.data + valid_set.data)
            torch.save(ontology_dict, './drop_ontology.pt')
    else:
        ontology = Path(args.ontology)
        if ontology.exists():
            ontology_dict = torch.load(ontology)
        else:
            ontology_dict = make_ontology(fn_list = Path(args.data_path).glob("*.txt"))
            torch.save(ontology_dict, './ontology.pt')

        dataset_name = f'./dataset_maxdepth{max_depth}_samplenum{sample_num}.pt'
        if Path(dataset_name).exists():
            dataset = torch.load(dataset_name)
        else:
            dataset = ClassificationDataset(root=hparams.data_path, ipc_len=ipc_len, sample_num=sample_num)
            torch.save(dataset, dataset_name)
    
        print('Total docs num: ', len(dataset))
    
        train_set_size = int(len(dataset) * 0.95)
        valid_set_size = len(dataset) - train_set_size

        train_set, valid_set = torch.utils.data.random_split(dataset, [train_set_size, valid_set_size], generator=seed)

    collate = hierarichical_classification_collate(label_list=ontology_dict['4_depth'], max_depth=max_depth)
    train_dataloader = DataLoader(train_set, num_workers=32, batch_size=8,
                            collate_fn=collate, shuffle=True, drop_last=True)
    valid_dataloader = DataLoader(valid_set, num_workers=32, batch_size=8,
                            collate_fn=collate, shuffle=False, drop_last=True)
    model = jjlClassifier(hparams, 
                          depth=max_depth, 
                          plm='tanapatentlm/patentdeberta_large_spec_128_pwi',
                          label_dict=collate.label2ipc,
                          )
    if is_test:
        ckpt = torch.load(hparams.cls_ckpt)['state_dict']
        print(model.load_state_dict(ckpt, strict=False))
        
    ckpt_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        filename="epoch_end_checkpoints-{epoch:02d}-{val_loss:.8f}",
        save_top_k=1,
        mode="min",
        save_last=True  # When ``True``, saves an exact copy of the checkpoint to a file `last.ckpt` whenever a checkpoint file gets saved.
    )

    ckpt_callback_steps = pl.callbacks.ModelCheckpoint(
        monitor="train_loss",
        every_n_train_steps = 1000,
        filename="intermediate_checkpoints-{epoch:02d}-{step:02d}-{train_loss:.8f}",
        save_top_k=1,
        mode="min",
        save_last=True
    )
    prediction_callback = CustomWriter()

    SWA = pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2)
    tblogger = TensorBoardLogger(save_dir=hparams.output_path, name="lightning_logs")
    device_cnt = torch.cuda.device_count()
    print("device count = {}".format(device_cnt))

    trainer = pl.Trainer(gpus=device_cnt,
                         max_epochs= hparams.epochs,
                         strategy="ddp" if device_cnt > 1 else None,
                         callbacks=[ckpt_callback, ckpt_callback_steps, prediction_callback],
                         logger=tblogger, 
                         gradient_clip_val=1.0,
                         val_check_interval=0.01,
                         accumulate_grad_batches=10,
                         num_sanity_val_steps=20)
    if not is_test: 
        trainer.fit(model, train_dataloader, valid_dataloader)
    else:
        trainer.validate(model=model, dataloaders=valid_dataloader)
        

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true') 
    parser.add_argument('--test', action='store_true') 
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    drop=True
    main(drop, args.test)
