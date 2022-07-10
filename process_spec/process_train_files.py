import numpy as np
import pandas as pd
import os
import re
import pickle
import torch
from transformers import AutoTokenizer
from multiprocessing import Pool
from tqdm.contrib.concurrent import process_map
import pickle

os.environ["TOKENIZERS_PARALLELISM"] = "true"

train_files = os.listdir("../storage/train_spec")

def process_train_text(idx):
    with open("../storage/train_spec/" + str(train_files[idx]), "r") as f:
        data = f.read()
    triplet = data.split('\n\n\n')
    q, p, n = triplet
    try:
        q_ttl = re.search("<TTL>([\s\S]*?)<IPC>", q).group(1)
        q_abst = re.search("<ABST>([\s\S]*?)<CLMS>", q).group(1)
        q_clms = re.search("<CLMS>([\s\S]*?)<DESC>", q).group(1)
        q = q_ttl + " " + q_abst + " " + q_clms

        p_ttl = re.search("<TTL>([\s\S]*?)<IPC>", p).group(1)
        p_abst = re.search("<ABST>([\s\S]*?)<CLMS>", p).group(1)
        p_clms = re.search("<CLMS>([\s\S]*?)<DESC>", p).group(1)
        p = p_ttl + " " + p_abst + " " + p_clms

        n_ttl = re.search("<TTL>([\s\S]*?)<IPC>", n).group(1)
        n_abst = re.search("<ABST>([\s\S]*?)<CLMS>", n).group(1)
        n_clms = re.search("<CLMS>([\s\S]*?)<DESC>", n).group(1)
        n = n_ttl + " " + n_abst + " " + n_clms

        processed_file = '\n\n\n'.join([q,p,n])
        with open("../storage/train_spec/" + str(train_files[idx]), "w") as f:
            f.write(processed_file)
    except:
        pass
    
    

if __name__ == "__main__":
    process_map(process_train_text, range(0, len(train_files)))
