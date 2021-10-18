import random
import collections
import torch
from enum import IntEnum
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class Token(IntEnum) :
    PAD = 0
    UNK = 1
    SOS = 2
    EOS = 3
    CLS = 4
    SEP = 5

class GptDataset(Dataset) :
    def __init__(self, idx_data, max_size) :
        super(GptDataset , self).__init__()
        self.idx_data = [idx_list[-max_size:] for idx_list in idx_data]
        self.max_size = max_size
        
    def __len__(self) :
        return len(self.idx_data)

    def __getitem__(self , idx) :
        return self.idx_data[idx]

class GptCollator:
    def __init__(self, len_data, batch_size, size_gap = 5):
        self.len_data = len_data
        self.batch_size = batch_size
        self.size_gap = size_gap
        self.data_size = len(len_data)
        
    def sample(self) :
        batch_map = collections.defaultdict(list)
        idx_list = []
        batch_index = []
    
        for idx in range(self.data_size) :
            len_idx = self.len_data[idx]
            len_group = len_idx // self.size_gap
            batch_map[len_group].append(idx)
            
        batch_key = list(batch_map.keys())
        batch_key = sorted(batch_key, key=lambda x : x, reverse=True) 
        for key in batch_key :
            idx_list.extend(batch_map[key])
    
        for i in range(0, self.data_size, self.batch_size) :
            batch_index.append(idx_list[i:i+self.batch_size])
    
        random.shuffle(batch_index)
        return batch_index
    
    def __call__(self, batch_samples):   
        batch_ids = []
        batch_pos = []
        for idx_list in batch_samples:
            batch_ids.append(torch.tensor(idx_list + [Token.PAD]))
            batch_pos.append(torch.arange(1,len(idx_list)+1))

        batch_ids_tensor = pad_sequence(batch_ids, batch_first=True, padding_value=Token.PAD)
        batch_pos_tensor = pad_sequence(batch_pos, batch_first=True, padding_value=Token.PAD)
        
        return {'in' : batch_ids_tensor[:,:-1], 
            'pos' : batch_pos_tensor, 
            'out' : batch_ids_tensor[:,1:]
        }
