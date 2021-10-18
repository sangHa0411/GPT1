import sys
import random
import re
import argparse
import multiprocessing
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from konlpy.tag import Mecab
from model import PaddingMask, LookAheadMask, TransformerDecoder

from dataset import *
from loader import *
from scheduler import *
from preprocessor import *

def progressLearning(value, endvalue, loss, acc, bar_length=50):
    percent = float(value + 1) / endvalue
    arrow = '-' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    sys.stdout.write("\r[{0}] {1}/{2} \t Loss : {3:.3f} , Acc : {4:.3f}".format(arrow + spaces, value+1, endvalue, loss, acc))
    sys.stdout.flush()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def train(args) :
    # -- Seed
    seed_everything(args.seed)

    # -- Device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- Data
    print('Load Raw Data')
    data = get_data(args.data_dir, args.file_size)

    print('Extract Text Data')
    text_data = []
    for json_data in tqdm(data) :
        text_list = preprocess_data(json_data)
        text_data.extend(text_list)

    # -- Tokenizer & Encoder
    sys.path.append('./Tokenizer')
    from tokenizer import *
    mecab = Mecab()
    sen_preprocessor = SenPreprocessor(mecab)
    tokenizer = get_spm(os.path.join(args.token_dir, 'tokenizer.model'))
    v_size = len(tokenizer)

    print('Encode Text Data')
    idx_data = []
    for text in tqdm(text_data) :
        text = sen_preprocessor(text)
        if text != None :
            idx_list = tokenizer.encode_as_ids(text)
            idx_data.append(idx_list)

    # -- Dataset
    dset = GptDataset(idx_data, args.max_size)
    dset_len = [len(data) for data in dset]

    print('Data Size : %d\n' %len(dset))

    # -- DataLoader
    collator = GptCollator(dset_len, args.batch_size)
    data_loader = DataLoader(dset,
        num_workers=multiprocessing.cpu_count()//2,
        batch_sampler=collator.sample(),
        collate_fn=collator
    )
   
    # -- Model
    # Masking
    padding_mask = PaddingMask()
    lookahead_mask = LookAheadMask(use_cuda)
    # Transformer Decoder
    gpt_1 = TransformerDecoder(
        layer_size=args.layer_size, 
        max_size=args.max_size, 
        v_size=v_size, 
        d_model=args.embedding_size,
        num_heads=args.head_size,
        hidden_size=args.hidden_size,
        drop_rate=0.1,
        norm_rate=1e-6,
        cuda_flag=use_cuda
    ).to(device)

    init_lr = 1e-4

    # -- Optimizer
    optimizer = optim.Adam(gpt_1.parameters(), 
        lr=init_lr, 
        betas=(0.9,0.98), 
        eps=1e-9,
        weight_decay=args.weight_decay
    )

    # -- Scheduler
    schedule_fn = Scheduler(args.embedding_size, init_lr, args.warmup_steps)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, 
        lr_lambda = lambda epoch: schedule_fn(epoch)
    )
    
    # -- Logging
    writer = SummaryWriter(args.log_dir)

    # -- Criterion 
    criterion = nn.CrossEntropyLoss().to(device)

    # -- Training
    log_count = 0
    # for each epoch
    for epoch in range(args.epochs) :
        idx = 0
        mean_loss = 0.0
        mean_acc = 0.0
        print('Epoch : %d/%d \t Learning Rate : %e' %(epoch, args.epochs, optimizer.param_groups[0]["lr"]))
        # training process
        for data in data_loader :
            optimizer.zero_grad()
            writer.add_scalar('learning_rate', optimizer.param_groups[0]["lr"], idx)

            in_data = data['in'].long().to(device)
            mask_data = padding_mask(in_data)
            mask_data = lookahead_mask(mask_data)

            label_data = data['out'].long().to(device)
            label_data = torch.reshape(label_data, (-1,))

            out_data = gpt_1(in_data, mask_data)
            out_data = torch.reshape(out_data, (-1,v_size))

            loss = criterion(out_data , label_data)
            acc = (torch.argmax(out_data, dim=-1) == label_data).float().mean()

            loss.backward()
            optimizer.step()
            scheduler.step()
        
            progressLearning(idx, len(data_loader), loss.item(), acc.item())
            mean_loss += loss
            mean_acc += acc

            if (idx + 1) % 100 == 0 :
                writer.add_scalar('train/loss', loss.item(), log_count)
                writer.add_scalar('train/acc', acc.item(), log_count)
                log_count += 1
            idx += 1

        mean_loss /= len(data_loader)
        mean_acc /= len(data_loader)

        torch.save({'epoch' : (epoch) ,  
            'model_state_dict' : gpt_1.state_dict() , 
            'loss' : mean_loss.item() , 
            'acc' : mean_acc.item()} , 
            f'./Model/checkpoint_gpt.pt') 

        print('\nMean Loss : %.3f \t Mean Accuracy : %.3f\n' %(mean_loss.item(), mean_acc.item()))
    
    
if __name__ == '__main__' :
    parser = argparse.ArgumentParser()

    # Training argument
    parser.add_argument('--seed', type=int, default=777, help='random seed (default: 777)')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train (default: 30)')
    parser.add_argument('--warmup_steps', type=int, default=2000, help='warmup steps of train (default: 2000)')
    parser.add_argument('--max_size', type=int, default=256, help='max size of sequence (default: 256)')
    parser.add_argument('--layer_size', type=int, default=12, help='layer size of model (default: 12)')
    parser.add_argument('--embedding_size', type=int, default=768, help='embedding size of token (default: 768)')
    parser.add_argument('--hidden_size', type=int, default=3072, help='hidden size of position-wise layer (default: 3072)')
    parser.add_argument('--head_size', type=int, default=12, help='head size of multi head attention (default: 12)')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training (default: 64)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay of optimizer (default: 1e-4)')

    # Container environment
    parser.add_argument('--file_size', type=int, default=10, help='size of newspaper file')
    parser.add_argument('--data_dir', type=str, default='./Data')
    parser.add_argument('--model_dir', type=str, default='./Model')
    parser.add_argument('--token_dir', type=str, default='./Tokenizer')
    parser.add_argument('--log_dir' , type=str , default='./Log')

    args = parser.parse_args()
    train(args)

