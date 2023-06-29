import os
import pickle
import random
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import math


from torch.nn import TransformerEncoder,TransformerEncoderLayer

device="cuda" if torch.cuda.is_available() else "cpu"

class Data(Dataset):
    def __init__(self, root, max_seq=2048, random_seq=True):
        self.root = root
        self.max_seq = max_seq
        self.random_seq = random_seq
        fs=[os.path.join(root, f) for f in os.listdir(self.root)]
        self.data_files = [f for f in fs if os.path.isfile(f)]
    def __len__(self):
        return len(self.data_files)
    def __getitem__(self, idx):
        i_stream = open(self.data_files[idx], "rb")
        # return pickle.load(i_stream), None
        raw_mid = torch.tensor(pickle.load(i_stream), 
                                   dtype=torch.long)
        i_stream.close()
        x,y = get_xy(raw_mid, self.max_seq, self.random_seq)
        return x,y

def get_xy(raw_mid, max_seq, random_seq, token_end=388):
    x = torch.full((max_seq, ), token_end+1, dtype=torch.long)
    y = torch.full((max_seq, ), token_end+1, dtype=torch.long)
    raw_len = len(raw_mid)
    full_seq = max_seq + 1 
    if(raw_len == 0):
        return x, y
    if(raw_len < full_seq):
        x[:raw_len] = raw_mid
        y[:raw_len-1] = raw_mid[1:]
        y[raw_len-1] = token_end
    else:
        if(random_seq):
            end_range = raw_len - full_seq
            start = random.randint(0, end_range)
        else:
            start = 0
        end = start + full_seq
        data = raw_mid[start:end]
        x = data[:max_seq]
        y = data[1:full_seq]
    return x, y



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout = 0.1, max_len = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)\
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)    






# d_model is embedding dimension
class Model(nn.Module):
    def __init__(self, ntoken, d_model, nhead, d_hid,
                 nlayers, dropout=0.1):
        super().__init__() 
        self.model_type="Transformer"
        self.pos_encoder=PositionalEncoding(d_model,dropout)
        encoder_layers = TransformerEncoderLayer(d_model,
                                 nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(
            encoder_layers, nlayers)
        self.embedding=nn.Embedding(ntoken,d_model)
        self.d_model=d_model
        self.linear=nn.Linear(d_model,ntoken)
        self.init_weights()
        
    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange,initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)        

    def forward(self,src):
        mask = nn.Transformer.generate_square_subsequent_mask(
            src.shape[1])        
        src=self.embedding(src)*math.sqrt(self.d_model)
        src=self.pos_encoder(src)
        output=self.transformer_encoder(src,mask)
        output=self.linear(output)
        return output


