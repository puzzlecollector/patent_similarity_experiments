import numpy as np 
import pandas as pd 
from sentence_transformers import SentenceTransformer, InputExample, losses, util, evaluation
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
import torch 
from torch.utils.data import DataLoader
import math
from datetime import datetime 
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

df = pd.read_excel("0814_라벨링세트_6주차_병합.xlsx") 

# list of checkpoints 
ckpts = ["tanapatentlm/patentdeberta_large_spec_128_pwi", 
         "Model/simcse-patent-for-bert", 
         "Model/simcse-DEBERTA", 
         "Model/dcpcse-patent-for-bert", 
         "anferico/bert-for-patents"] 

df = df.loc[df['라벨링'].notnull(), ['쿼리 문장', '후보 문장', '라벨링']] 
df = df.dropna() 
# fix incorrect labels 
labels_fixed = [] 
labels = df['라벨링'].values 

for i in range(len(labels)):
    if labels[i] == 0.1:
        labels_fixed.append(1.0) 
    elif labels[i] not in [0, 0.5, 0.8, 1.0]:
        labels_fixed.append(None) 
    else:
        labels_fixed.append(labels[i]) 
        
df['라벨링'] = labels_fixed 
df = df.dropna() 

for idx in range(len(ckpts)):
    if idx < 2:
        continue 
    print("========== Training model {} ==========".format(ckpts[idx])) 
    tokenizer = AutoTokenizer.from_pretrained(ckpts[0]) 
    queries, candidates = df['쿼리 문장'].values, df['후보 문장'].values 
    lengths = [] 
    for i in tqdm(range(df.shape[0])): 
        input_ids = tokenizer(queries[i], candidates[i])['input_ids'] 
        lengths.append(len(input_ids))
    model = SentenceTransformer(ckpts[idx])
    model.max_seq_length = 128
    
    full_x = df[['쿼리 문장', '후보 문장']].values 
    full_y = df['라벨링'].values 
    cat_y = [] 
    cat_dict = {0:0, 0.5:1, 0.8:2, 1.0:3} 
    for i in range(len(full_y)):
        cat_y.append(cat_dict[full_y[i]]) 
    
    train_x, test_x, train_y, test_y = train_test_split(full_x, full_y, test_size=0.2, random_state=42, stratify=cat_y) 
    print(train_x.shape, test_x.shape, train_y.shape, test_y.shape) 
    
    num_epochs=3
    model = SentenceTransformer(ckpts[0]) 
    train_examples = [] 
    for i in range(len(train_x)):
        train_examples.append(InputExample(texts=[train_x[i][0],train_x[i][1]], label=float(train_y[i])))

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32) 
    train_loss = losses.CosineSimilarityLoss(model=model) 

    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) # 10% of train data for warmup 

    if idx == 0: 
        name = "simcse_patent_for_bert"
    elif idx == 1:
        name = "simcse_DEBERTA" 
    elif idx == 2:
        name = "dcpcse_patent_for_bert" 
    elif idx == 3:
        name = "patentdeberta_large_spec_128_pwi" 
    elif idx == 4:
        name = "bert_for_patents" 


    model.fit(train_objectives=[(train_dataloader, train_loss)],
              epochs=num_epochs, 
              warmup_steps=warmup_steps, 
              output_path="../storage/" + name) 
    
    print("Training Done!") 
    print("Evaluating...") 
    eval_s1, eval_s2, scores = [], [], [] 
    for i in range(len(test_x)):
        eval_s1.append(test_x[i][0]) 
        eval_s2.append(test_x[i][1]) 
        scores.append(float(test_y[i])) 

    evaluator = evaluation.EmbeddingSimilarityEvaluator(eval_s1, eval_s2, scores)  
    pearson_correlation = model.evaluate(evaluator)  
    print("test pearson_correlation: {}".format(pearson_correlation))
    
    # calculate accuracy 
    cnt = 0 
    for i in tqdm(range(len(test_x))): 
        emb1 = model.encode(test_x[i][0], convert_to_tensor=True) 
        emb2 = model.encode(test_x[i][1], convert_to_tensor=True) 
        cosine_scores = util.cos_sim(emb1, emb2) 
        if test_y[i] == 0 and cosine_scores < 0.5:
            cnt += 1 
        elif test_y[i] == 0.5 and (cosine_scores >= 0.5 and cosine_scores < 0.8):
            cnt += 1 
        elif test_y[i] >= 0.8 and cosine_scores >= 0.8: 
            cnt += 1 

    print("accuracy = {}%".format(cnt / len(test_y) * 100.0))


