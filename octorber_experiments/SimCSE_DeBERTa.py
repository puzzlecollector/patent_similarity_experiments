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

dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=8, name="sts-dev") 
        
corpus_lines = [] 

L = 100 

try: 
    file_name = "../storage/kr_corpus_v1.0.txt"
    print(f'File Size is {os.stat(file_name).st_size / (1024 * 1024)} MB')
    txt_file = open(file_name)
    count = 0
    for line in tqdm(txt_file, position=0, leave=True):
        # we can process file line by line here, for simplicity I am taking count of lines
        if count % 100 == 0 and len(line) > L: 
            corpus_lines.append(line)
        
        count += 1
        
        if len(corpus_lines) == 1000000:
            break
    txt_file.close()
except KeyboardInterrupt: 
    print("keyboard interrupted - exiting...") 
        
print(f"total line count {count}")

word_embedding_model = models.Transformer("tanapatentlm/patent-ko-deberta", max_seq_length=128)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension()) 
model = SentenceTransformer(modules=[word_embedding_model, pooling_model]) 

train_samples = [] 

for i in tqdm(range(len(corpus_lines)), desc="forming training data"): 
    train_samples.append(InputExample(texts=[corpus_lines[i], corpus_lines[i]])) 

model_save_path = "Ko_Patent_DeBERTa_SimCSE" 

train_batch_size = 32
num_epochs = 1 

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size = train_batch_size) 
train_loss = losses.MultipleNegativesRankingLoss(model) 

warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) 

model.fit(train_objectives = [(train_dataloader, train_loss)], 
          evaluator = dev_evaluator,
          epochs = num_epochs, 
          evaluation_steps=1000, 
          warmup_steps = warmup_steps, 
          output_path = model_save_path) 

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

model = SentenceTransformer(model_save_path) 
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, batch_size = train_batch_size, name="sts-test") 
test_evaluator(model, output_path = model_save_path) 


