import random
import collections
from dataset import Token
import torch
from torch.nn.utils.rnn import pad_sequence

class Collator:
    def __init__(self, len_data, batch_size, size_gap=5):
        self.len_data = len_data
        self.size_gap = size_gap
        self.batch_size = batch_size
        self.data_size = len(len_data)
        
    def sample(self) :
        batch_map = collections.defaultdict(list)
        idx_list = []
        batch_index = []
    
        for idx in range(self.data_size) :
            len_group = self.len_data[idx]//self.size_gap            
            batch_map[len_group].append(idx)
            
        batch_key = list(batch_map.keys())
        batch_key = sorted(batch_key, key=lambda x : x, reverse=True) 
        # sorting idx list based on size group
        for key in batch_key :
            idx_list.extend(batch_map[key])
    
        # slicing batch_size
        for i in range(0, self.data_size, self.batch_size) :
            batch_index.append(idx_list[i:i+self.batch_size])
    
        random.shuffle(batch_index)
        return batch_index
    
    def __call__(self, batch_samples):   
        tensor_list = []
        for idx_list in batch_samples:
            tensor_list.append(torch.tensor(idx_list+[Token.PAD]))
        tensor_data = pad_sequence(tensor_list, batch_first=True, padding_value=Token.PAD)

        return {'input' : tensor_data[:,:-1], 'label' : tensor_data[:,1:]}
