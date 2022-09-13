from torch.utils.data import Dataset
from pathlib import Path
import torch
import re
from typing import List
class TripletData(Dataset):
    """Patent document as txt file"""
    def __init__(self, root: Path, is_debug=False):
        super().__init__()
        self.data = []
        if is_debug:
            with (root / "test_triplet.csv").open("r", encoding="utf8") as f:
                for i, triplet in enumerate(f):
                    if i >= 100000: break  # pylint: disable=multiple-statements
                    query, positive, negative = triplet.strip().split(",")
                    data = []
                    data.append(root / f"{query}.txt")
                    data.append(root / f"{positive}.txt")
                    data.append(root / f"{negative}.txt")
                    self.data.append(data)
        else:
            for fn in root.glob("*.txt"):
                self.data.append([fn])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class custom_collate(object):
    def __init__(self, tokenizer, is_debug):
        self.tokenizer = tokenizer
        self.chunk_size = 256
        self.debug = is_debug

    def load_file(self, fn_list: List[Path]):
        """Load file and read contents"""
        triplet = []
        for fn in fn_list:
            with fn.open("r", encoding="utf8") as f:
                content = f.read()
                triplet.append(content)
        return triplet

    def clean_text(self, t):
        x = re.sub("\d+.", "", t)
        x = x.replace("\n", " ")
        x = x.strip()
        return x

    def encode_document(self, doc_list):
        text_input = []
        for doc in doc_list:
            transformed_text = self.transform_document(doc)
            text_input.append(transformed_text)
        encoded_doc = self.tokenizer(text_input, max_length=self.chunk_size, 
                padding='max_length', truncation=True, return_tensors='pt', )
        return encoded_doc
    
    def transform_document(self, doc):
        ttl = re.search("<TTL>([\s\S]*?)<IPC>", doc).group(1)
        ipc = re.search("<IPC>([\s\S]*?)<ABST>", doc).group(1)
        clms = re.search("<CLMS>([\s\S]*?)<DESC>", doc).group(1)
        ttl = ttl.lower()  # convert title to lower case
        ipc = ipc[:3]  # get first three characters
        # get first claim as long as it is not canceled
        ind_clms = clms.split('\n\n')
        selected_clm = ind_clms[0]
        for ind_clm in ind_clms:
            if '(canceled)' in ind_clm:
                continue
            else:
                selected_clm = ind_clm
                break
        selected_clm = self.clean_text(selected_clm)
        text_input = ipc + " " + ttl + self.tokenizer.sep_token + selected_clm
        return text_input

    def __call__(self, batch):

        if self.debug:
            triplet = zip(*list(map(self.load_file, batch)))
            q,p,n = triplet
            assert len(q) == len(p) == len(n)
            encoded_q = self.encode_document(q)
            encoded_p = self.encode_document(p)
            encoded_n = self.encode_document(n)
            return {"q" : encoded_q,
                    "p" : encoded_p,
                    "n" : encoded_n}
        else:
            q = list(map(self.load_file, batch))
            q = [qq[0] for qq in q]
            encoded_q = self.encode_document(q)
            return {"q" : encoded_q}
