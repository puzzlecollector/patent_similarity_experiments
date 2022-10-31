class NeuralClf(pl.LightningModule):
    def __init__(self, hparams=dict(), plm="tanapatentlm/patent-ko-deberta"):
        super(NeuralClf, self).__init__()
        self.config = AutoConfig.from_pretrained(plm)
        self.model = AutoModel.from_pretrained(plm, config=self.config)
        self.tokenizer = AutoTokenizer.from_pretrained(plm)
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["[IPC]", "[TTL]", "[CLMS]", "[DESC]"]}) 
        self.mean_pooler = MeanPooling()
        self.fc = nn.Linear(self.config.hidden_size, 2)
        self._init_weights(self.fc)
        self.multi_dropout = MultiSampleDropout(0.2, 8, self.fc)
        self.metric = nn.CrossEntropyLoss() 
        '''
        if "additional_special_tokens" in self.hparams and self.hparams["additional_special_tokens"]:
            print("="*30 + " adding special tokens " + "="*30) 
            additional_special_tokens = self.hparams["additional_special_tokens"] 
            self.tokenizer.add_special_tokens({"additional_special_tokens":additional_special_tokens})
            self.model.resize_token_embeddings(len(self.tokenizer)) 
        ''' 
        self.model.resize_token_embeddings(len(self.tokenizer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attn_masks):
        x = self.model(input_ids, attn_masks)[0] 
        x = self.mean_pooler(x, attn_masks) 
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
