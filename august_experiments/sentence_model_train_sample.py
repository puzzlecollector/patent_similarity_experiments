import numpy as np 
import pandas as pd 
from sentence_transformers import SentenceTransformer, InputExample, losses, util, evaluation
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
import torch 
from torch.utils.data import DataLoader
import math
from datetime import datetime
 
df1 = pd.read_excel("0710_라벨링세트_1주차_병합.xlsx") 
df2 = pd.read_excel("0717_라벨링세트_2주차_병합.xlsx") 
df3 = pd.read_excel("0724_라벨링세트_4주차_병합.xlsx") 
df4 = pd.read_excel("0724_라벨링세트_3주차_병합.xlsx") 

df = pd.concat([df1, df2, df3, df4], axis=0) 

df = df.loc[df['라벨링'].notnull(), ['쿼리 문장', '후보 문장', '라벨링']] 

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

train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

full_x = train_df[['쿼리 문장', '후보 문장']].values
full_y = train_df['라벨링'].values 

test_x = test_df[['쿼리 문장', '후보 문장']].values
test_y = test_df['라벨링'].values   

cat_y = [] 
cat_dict = {0:0, 0.5:1, 0.8:2, 1.0:3} 

for i in range(len(full_y)):
    cat_y.append(cat_dict[full_y[i]])
        
        
skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True) 

for idx, (train_idx, valid_idx) in enumerate(skf.split(full_x, cat_y)):
    if idx > 1:
        break 
    print("==================== KFOLD {} ====================".format(idx+1))
    num_epochs = 3 
    model = SentenceTransformer("princeton-nlp/sup-simcse-roberta-large")  
    train_x, valid_x = full_x[train_idx], full_x[valid_idx] 
    train_y, valid_y = full_y[train_idx], full_y[valid_idx] 
    
    train_examples = [] 
    for i in range(len(train_x)):
        train_examples.append(InputExample(texts=[train_x[i][0], train_x[i][1]], label=float(train_y[i])))
    
    eval_s1, eval_s2, scores = [], [], [] 
    for i in range(len(valid_x)):
        eval_s1.append(valid_x[i][0]) 
        eval_s2.append(valid_x[i][1]) 
        scores.append(float(valid_y[i])) 
    
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.CosineSimilarityLoss(model=model)
    
    evaluator = evaluation.EmbeddingSimilarityEvaluator(eval_s1, eval_s2, scores) 
    
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) # 10% of train data for warmup
    
    model.fit(train_objectives=[(train_dataloader, train_loss)], 
              evaluator=evaluator, 
              epochs=num_epochs,
              evaluation_steps=600,
              warmup_steps=warmup_steps,
              output_path='../storage/sup_simcse_roberta_large_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    
