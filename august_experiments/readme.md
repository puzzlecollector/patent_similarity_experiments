Document Ranker Experiments being conducted for the month of 2022 August. 

The approximate list is as follows 

1. PARADE where inputs are independent claims of the document, using a dimension size of (b, 6, 256) corresponding to at most 6 independent claims, each having a limitation of 256 tokens 

2. deberta base spec pre-trained model trained on only the first claims, with continual learning from FirstP (512) ckpt for faster convergence 

3. same model as 2 but with cross entropy loss based on cosine similarity   

4. patent bert large pre-trained model for FirstP, but with independent claims cleaned and separated via the [SEP] token 
