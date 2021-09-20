import sys
import random
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
from model import TransformerDecoder
from dataset import *
from collator import *
from loader import *
from preprocessor import *

def schedule_fn(epoch, d_model, init_lr, warmup_steps) :
    step_num = epoch + 1
    val1 = d_model ** (-0.5)
    arg1 = step_num ** (-0.5)
    arg2 = (warmup_steps ** (-1.5)) * step_num
    val2 = min(arg1 , arg2) 
    return (val1 * val2) / init_lr

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
    data = get_data(args.data_dir)

    print('Extract Text Data')
    text_data = []
    for json_data in tqdm(data) :
        text_list = preprocess_data(json_data)
        text_data += text_list

    # -- Tokenizer & Encoder
    kor_tokenizer = get_spm(args.token_dir, 'kor_tokenizer.model')
    kor_v_size = len(kor_tokenizer)

    print('Encode Text Data')
    idx_data = []
    for text in tqdm(text_data) :
        idx_list = kor_tokenizer.encode_as_ids(text)
        idx_data.append(idx_list)

    # -- Dataset
    dset = GptDataset(idx_data, args.max_size)
    dset_len = [len(data) for data in dset]

    # -- DataLoader
    collator = GptCollator(dset_len, args.batch_size)
    data_loader = DataLoader(dset,
        num_workers=multiprocessing.cpu_count()//2,
        batch_sampler=collator.sample(),
        collate_fn=collator
    )
   
    # -- Model
    # Transformer Decoder
    gpt_1 = TransformerDecoder(layer_size=args.layer_size, 
        max_size=args.max_size, 
        v_size=kor_v_size, 
        d_model=args.embedding_size,
        num_heads=args.head_size,
        hidden_size=args.hidden_size,
        drop_rate=0.1,
        norm_rate=1e-6,
        cuda_flag=use_cuda
    ).to(device)

    # -- Optimizer
    optimizer = optim.Adam(gpt_1.parameters(), lr=1e-4, betas=(0.9,0.98), eps=1e-9)

    # -- Scheduler
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, 
        lr_lambda = lambda epoch: schedule_fn(epoch = epoch,
            d_model = args.embedding_size, 
            init_lr = 1e-4, 
            warmup_steps=args.warmup_steps)
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
            in_data = data['in'].long().to(device)
            label_data = data['out'].long().to(device)
            label_data = torch.reshape(label_data, (-1,))

            optimizer.zero_grad()
        
            out_data = gpt_1(in_data)
            out_data = torch.reshape(out_data, (-1,kor_v_size+1))

            loss = criterion(out_data , label_data)
            acc = (torch.argmax(out_data, dim=-1) == label_data).float().mean()

            loss.backward()
            optimizer.step()
        
            progressLearning(idx, len(data_loader), loss.item(), acc.item())
            mean_loss += loss
            mean_acc += acc

            if (idx + 1) % 10 == 0 :
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

        scheduler.step()
        print('\nMean Loss : %.3f \t Mean Accuracy : %.3f\n' %(mean_loss.item(), mean_acc.item()))
    
    
if __name__ == '__main__' :
    parser = argparse.ArgumentParser()

    # Training argument
    parser.add_argument('--seed', type=int, default=777, help='random seed (default: 777)')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
    parser.add_argument('--warmup_steps', type=int, default=2000, help='warmup steps of train (default: 2000)')
    parser.add_argument('--max_size', type=int, default=128, help='max size of sequence (default: 128)')
    parser.add_argument('--layer_size', type=int, default=6, help='layer size of model (default: 6)')
    parser.add_argument('--embedding_size', type=int, default=512, help='embedding size of token (default: 512)')
    parser.add_argument('--hidden_size', type=int, default=2048, help='hidden size of position-wise layer (default: 2048)')
    parser.add_argument('--head_size', type=int, default=8, help='head size of multi head attention (default: 8)')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training (default: 128)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default='./Data/Version1.0')
    parser.add_argument('--model_dir', type=str, default='./Model')
    parser.add_argument('--token_dir', type=str, default='./Token')
    parser.add_argument('--log_dir' , type=str , default='./Log')

    args = parser.parse_args()
    train(args)
