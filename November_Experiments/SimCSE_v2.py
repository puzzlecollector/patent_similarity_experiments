from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import math
from sentence_transformers import models, losses
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import logging
from datetime import datetime
import os
import gzip
import csv
import pandas as pd
import re
import pickle 
​
​
# create train data 
files = os.listdir("../storage/kr_triplet_v2.1") 
print(len(files)) 
try: 
    lines = [] 
​
    L = 50 
​
    for f in tqdm(files[:1000]): 
        with open("../storage/kr_triplet_v2.1/" + str(f), "r") as f: 
            data = f.read() 
            for split in re.split(r"wherein|[;\n]+", data.replace(".",";").replace("[CLMS]","").replace("[DESC]","")):
                if len(split) > L:
                    split = re.sub(r"\[IPC\](.*?)\[TTL\]", "", split) 
                    lines.append(split) 
except KeyboardInterrupt: 
    print("stop calculating...") 
    
with open('simcse_train.pkl','wb') as f:
    pickle.dump(lines, f)
​
​
# load validation data 
# create dev dataset 
dev_samples = [] 
with open("sts-dev.tsv") as file: 
    tsv_file = csv.reader(file, delimiter="\t") 
    for idx, line in enumerate(tsv_file): 
        if idx > 0:
            try: 
                score, sentence1, sentence2 = line[4], line[5], line[6] 
                score = float(score) / 5.0 
                dev_samples.append(InputExample(texts=[str(sentence1), str(sentence2)], label=score)) 
            except: 
                continue
​
dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=8, name="sts-dev") 
​
# load train data 
print("loading train data...") 
with open('simcse_train.pkl','rb') as f:
    corpus_lines = pickle.load(f)
print("done!") 
​
# define model  
word_embedding_model = models.Transformer("tanapatentlm/patent-ko-deberta", max_seq_length=256)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension()) 
model = SentenceTransformer(modules=[word_embedding_model, pooling_model]) 
​
print(model) 
    
train_samples = [] 
​
for i in tqdm(range(len(corpus_lines)), desc="forming training data"): 
    train_samples.append(InputExample(texts=[corpus_lines[i], corpus_lines[i]])) 
​
model_save_path = "Kor_SimCSE" 
​
train_batch_size = 16
num_epochs = 3
​
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size = train_batch_size) 
train_loss = losses.MultipleNegativesRankingLoss(model) 
​
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) 
​
model.fit(train_objectives = [(train_dataloader, train_loss)], 
          evaluator = dev_evaluator,
          epochs = num_epochs, 
          evaluation_steps=1000, 
          warmup_steps = warmup_steps, 
          output_path = model_save_path) 
​
# test model 
test_samples = [] 
with open("sts-test.tsv") as file: 
    tsv_file = csv.reader(file, delimiter="\t") 
    for idx, line in enumerate(tsv_file): 
        if idx > 0:
            try: 
                score, sentence1, sentence2 = line[4], line[5], line[6] 
                score = float(score) / 5.0 
                test_samples.append(InputExample(texts=[str(sentence1), str(sentence2)], label=score)) 
            except: 
                continue
​
model = SentenceTransformer(model_save_path) 
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, batch_size = train_batch_size, name="sts-test") 
test_evaluator(model, output_path = model_save_path) 
