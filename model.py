
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class LookAheadMask(nn.Module) :
    def __init__(self, cuda_flag) :
        super(LookAheadMask, self).__init__() 
        self.cuda_flag = cuda_flag
        
    def get_mask(self, sen_size) :
        mask_array = 1 - np.tril(np.ones((sen_size,sen_size)) , 0)
        mask_tensor = torch.tensor(mask_array , dtype = torch.float32 , requires_grad=False)
        mask_tensor = mask_tensor.unsqueeze(0)
        return mask_tensor
    
    def padding_mask(self, in_tensor) :
        batch_size, seq_size = in_tensor.shape
        flag_tensor = torch.where(in_tensor == 0.0 , 1.0 , 0.0)
        flag_tensor = torch.reshape(flag_tensor , (batch_size, 1, 1, seq_size)) 
        return flag_tensor, seq_size

    def forward(self, in_tensor,) :
        pad_mask, sen_size = self.padding_mask(in_tensor)
        lookahead_mask = self.get_mask(sen_size)
        if self.cuda_flag :
            lookahead_mask = lookahead_mask.cuda() 
        lookahead_mask = torch.maximum(pad_mask, lookahead_mask)
        return lookahead_mask

# Multihead Attention Layer
class MultiHeadAttention(nn.Module) :
    def __init__(self, d_model, num_heads) :
        super(MultiHeadAttention , self).__init__()
        self.d_model = d_model # vector size 
        self.num_heads = num_heads # head_size
        self.depth = int(d_model / num_heads)

        self.q_layer = nn.Linear(d_model, d_model)
        self.k_layer = nn.Linear(d_model, d_model)
        self.v_layer = nn.Linear(d_model, d_model)
        self.o_layer = nn.Linear(d_model, d_model)

        self.scale = torch.sqrt(torch.tensor(self.depth , dtype=torch.float32 , requires_grad=False))

    # tensor shape : (batch_size, sen_size, d_model)
    def split(self, tensor) :
        sen_size = tensor.shape[1]
        tensor = torch.reshape(tensor, (-1, sen_size, self.num_heads, self.depth))     
        tensor = tensor.permute(0,2,1,3) # (batch_size, num_heads, sen_size, depth)
        return tensor

    # tensor shape : (batch_size, num_heads, sen_size, dk)
    def merge(self, tensor) :
        sen_size = tensor.shape[2]
        tensor = tensor.permute(0,2,1,3) # (batch_size, sen_size, num_heads, depth)
        tensor = torch.reshape(tensor, (-1, sen_size, self.d_model)) #(batch_size, sen_size, embedding_dim)
        return tensor
    
    # scaled dot production
    def scaled_dot_production(self, q_tensor, k_tensor, v_tensor, m_tensor) :
        q_tensor = self.split(q_tensor)
        k_tensor = self.split(k_tensor)
        v_tensor = self.split(v_tensor)
        
        k_tensor_T = k_tensor.permute(0,1,3,2) # (batch_size, num_heads, depth, sen_size)

        qk_tensor = torch.matmul(q_tensor , k_tensor_T) # (batch_size, num_heads, sen_size, sen_size)
        qk_tensor /= self.scale

        if m_tensor != None :
            qk_tensor -= (m_tensor*1e+6)
            
        qk_tensor = F.softmax(qk_tensor, dim=-1)
        att = torch.matmul(qk_tensor, v_tensor) # (batch_size, num_heads, sen_size, depth)

        return att

    def forward(self, q_in, k_in, v_in, m_in) :
        q_tensor = self.q_layer(q_in)
        k_tensor = self.k_layer(k_in)
        v_tensor = self.v_layer(v_in)
        
        att_tensor = self.scaled_dot_production(q_tensor , k_tensor , v_tensor , m_in)
        multi_att_tensor = self.merge(att_tensor)
        
        o_tensor = self.o_layer(multi_att_tensor)
        return o_tensor

# Feedforward layer
class FeedForward(nn.Module) :
    def __init__(self, hidden_size, d_model) :
        super(FeedForward , self).__init__()
        self.hidden_size = hidden_size
        self.d_model = d_model
        # relu activation and input, output dim are same
        self.ff = nn.Sequential(nn.Linear(d_model , hidden_size), 
                                nn.ReLU(),
                                nn.Linear(hidden_size , d_model))

    def forward(self , in_tensor) :
        o_tensor = self.ff(in_tensor)
        return o_tensor

# Transformer Decoder Block
class DecoderBlock(nn.Module) :
    def __init__(self, d_model, num_heads, hidden_size, drop_rate, norm_rate) :
        super(DecoderBlock, self).__init__()
        # multihead attention layer & feedforward layer
        self.mha_layer = MultiHeadAttention(d_model , num_heads)
        self.ff_layer = FeedForward(hidden_size , d_model)
        # dropout layer & layer normalization layer
        self.drop1_layer = nn.Dropout(drop_rate)
        self.norm1_layer = nn.LayerNorm(d_model, eps=norm_rate)
        self.drop2_layer = nn.Dropout(drop_rate)
        self.norm2_layer = nn.LayerNorm(d_model, eps=norm_rate)
        
    def forward(self, in_tensor, mask_tensor) :
        # masked multihead attention sub layer
        # query : in_tensor, key : in_tensor, value : in_tensor, mask ; look ahead mask
        mha_tensor = self.mha_layer(in_tensor, in_tensor, in_tensor, mask_tensor)
        mha_tensor = self.drop1_layer(mha_tensor)
        h_tensor = self.norm1_layer(in_tensor+mha_tensor)
        # feedforward sub layer
        ff_tensor = self.ff_layer(h_tensor)
        ff_tensor = self.drop2_layer(ff_tensor)
        o_tensor = self.norm2_layer(h_tensor+ff_tensor)
        
        return o_tensor

# Transformer Decoder
class TransformerDecoder(nn.Module) :
    def __init__(self, layer_size, max_size, v_size, d_model, num_heads, hidden_size, drop_rate, norm_rate) :
        super(TransformerDecoder , self).__init__()
        self.layer_size = layer_size
        self.max_size = max_size
        self.v_size = v_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.drop_rate = drop_rate
        self.norm_rate = norm_rate
        
        self.em = nn.Embedding(num_embeddings=v_size, embedding_dim=d_model, padding_idx=0) # embedding
        self.pos_em = nn.Embedding(num_embeddings=max_size+1, embedding_dim=d_model, padding_idx=0) # positional embedding
        
        self.de_blocks = nn.ModuleList()
        self.drop_layer = nn.Dropout(drop_rate)
        self.o_layer = nn.Linear(d_model, v_size)
        for i in range(layer_size) :
            self.de_blocks.append(DecoderBlock(d_model, num_heads, hidden_size, drop_rate, norm_rate))
    
        self.init_param()
        
    def init_param(self) :
        for p in self.parameters() :
            if p.dim() > 1 :
                nn.init.xavier_uniform_(p)

    def get_feature(self, id_tensor, pos_tensor, mask_tensor) :
        # decoder input tensor
        em = self.em(id_tensor) # embedding
        pos = self.pos_em(pos_tensor) # positional embedding

        encoded = em+pos
        encoded = self.drop_layer(encoded) # dropout layer
    
        tensor_ptr = encoded
        for i in range(self.layer_size) :
            tensor_ptr = self.de_blocks[i](tensor_ptr, mask_tensor)
        return tensor_ptr
   
    def forward(self, id_tensor, pos_tensor, mask_tensor) :
        feature_tensor = self.get_feature(id_tensor, pos_tensor, mask_tensor)
        o_tensor = self.o_layer(tensor_ptr)
        return o_tensor
