import os 
import numpy as np 
import pandas as pd
import shutil
import pickle 
from tqdm import tqdm 

files = os.listdir("../storage/FGH_spec_ind_claims_triplet_v0.3.1") 

train_files = files[:900000] 
valid_files = files[900000:950000]
test_files = files[950000:] 

for i in tqdm(range(len(train_files)), desc="Splitting for train files"):
    with open("../storage/FGH_spec_ind_claims_triplet_v0.3.1/" + train_files[i], "r") as f:
        data = f.read() 
    with open("../storage/train_spec/" + train_files[i], "w") as f:
        f.write(data)  
        
for i in tqdm(range(len(valid_files)), desc="Splitting for valid files"):
    with open("../storage/FGH_spec_ind_claims_triplet_v0.3.1/" + valid_files[i], "r") as f:
        data = f.read() 
    with open("../storage/valid_spec/" + valid_files[i], "w") as f: 
        f.write(data)  
        
for i in tqdm(range(len(test_files)), desc="Splitting for test files"): 
    with open("../storage/FGH_spec_ind_claims_triplet_v0.3.1/" + test_files[i], "r") as f: 
        data = f.read() 
    with open("../storage/test_spec/" + test_files[i], "w") as f: 
        f.write(data) 
    
print("done!") 
