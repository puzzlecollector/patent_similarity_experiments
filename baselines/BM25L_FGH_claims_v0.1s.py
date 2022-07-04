import numpy as np 
import pandas as pd
import os 
from rank_bm25 import BM25Okapi, BM25L, BM25Plus
from tqdm import tqdm
from multiprocessing import Pool 
from tqdm.contrib.concurrent import process_map 
import pickle


test_files = os.listdir("FGH_claim_triplet_v0.1s/test") 

queries = [] 
positive, negative = [],[] 

for i in tqdm(range(len(test_files)), desc="processing test file"):
    with open("FGH_claim_triplet_v0.1s/test/" + test_files[i], 'r', encoding='utf-8') as f:
        data = f.read() 
    triplet = data.split('\n\n\n') 
    queries.append(triplet[0]) 
    positive.append(triplet[1])
    negative.append(triplet[2]) 
    
corpus = positive + negative 
tokenized_corpus = [doc.split(" ") for doc in tqdm(corpus, position=0, leave=True, desc="tokenizing doc")]

# indexing 
print("indexing tokenized corpus") 
bm25 = BM25L(tokenized_corpus) 
print("done!") 
print("calculating BM25 ranks") 

def process_bm25(idx):
    tokenized_query = queries[idx].split(" ")
    doc_scores = bm25.get_scores(tokenized_query)
    doc_ranks = doc_scores.argsort()[::-1] 
    for j in range(len(doc_ranks)):
        if idx == doc_ranks[j]:
            return idx, j+1 # returns (id
        

if __name__ == '__main__':
    r = process_map(process_bm25, range(0,10)) 

print("done! saving BM25 rank array") 
with open("bm25L_rank.pkl", "wb") as f:
    pickle.dump(r, f) 
    
print("calculating MRR") 
with open("bm25L_rank.pkl", "rb") as f:
    data = pickle.load(f) 

result = pd.DataFrame()
ranks = [data[i][1] for i in range(len(data))] 
result['rank'] = ranks 
result['r_rank'] = 1/result['rank']
total_count = result.count()['rank']
for i, r in enumerate([1,3,5,10,20,30,50,100]):
    subset = result.apply(lambda x : x['r_rank'] if int(x['rank']) <= r else 0, axis=1) 
    mrr = subset.sum() 
    mrr_count = subset.astype(bool).sum()
    print(f'MRR@{r}:', mrr / total_count, '/ count:', mrr_count)

print('average rank {}'.format(result['rank'].sum() / total_count)) 


    
