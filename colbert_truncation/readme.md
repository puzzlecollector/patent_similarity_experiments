## ColBERT with truncation (first 512 tokens) 

#### Procedure  
1. [Preprocess json data to form triplets](https://github.com/puzzlecollector/patent_similarity_experiments/blob/main/colbert_truncation/form_claims_triplets.ipynb)

2. [Train ColBERT with triplet data using first 512 tokens](https://github.com/puzzlecollector/patent_similarity_experiments/blob/main/colbert_truncation/train_truncation.ipynb) 

3. [Form collections and queries dataframes for inference](https://github.com/puzzlecollector/patent_similarity_experiments/blob/main/colbert_truncation/form_collections_and_queries.ipynb)

4. [Index, then create FAISS indexing, then retrieve and rerank top 1000 documents per query](https://github.com/puzzlecollector/patent_similarity_experiments/blob/main/colbert_truncation/inference_truncation.ipynb) - Due to time and memory limitations, I have only processed the first 100 queries. 


#### Experiment Results  
Trained with ~310,000 triplets, tested on 100 queries: **MRR = 0.208**

Trained with ~350,000 triplets (train+val combined), tested on 100 queries: **MRR = 0.212** 


#### Validating on a single query example 
[code for validating on a single query](https://github.com/puzzlecollector/patent_similarity_experiments/blob/main/colbert_truncation/single_query_score_calculation_example.ipynb) 
