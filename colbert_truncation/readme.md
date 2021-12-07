## ColBERT with truncation (first 512 tokens) 

#### Procedure  
1. [Preprocess json data to form triplets](https://github.com/puzzlecollector/patent_similarity_experiments/blob/main/colbert_truncation/form_claims_triplets.ipynb)

2. [Train ColBERT with triplet data using first 512 tokens](https://github.com/puzzlecollector/patent_similarity_experiments/blob/main/colbert_truncation/train_truncation.ipynb) 

3. [Form collections and queries dataframes for inference](https://github.com/puzzlecollector/patent_similarity_experiments/blob/main/colbert_truncation/form_collections_and_queries.ipynb)

#### Validating on a single query example 
[code for validating on a single query](https://github.com/puzzlecollector/patent_similarity_experiments/blob/main/colbert_truncation/single_query_score_calculation_example.ipynb) 
