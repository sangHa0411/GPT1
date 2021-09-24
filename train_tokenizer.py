import os
import re
import argparse
import kss
from tqdm import tqdm
from tokenizer import *
from loader import *

def preprocess_kor(sen) :
    sen = re.sub('[^가-힣0-9 \',.!?]' , '', sen)
    sen = re.sub(' {2,}' , ' ' , sen)
    return sen

def train(args) :
    print('Get Newspaper Data')
    data = get_data(args.data)
    
    text_path = os.path.join(args.dir, args.text)
    print('Extract Text Data')
    text_data = []
    for json_data in tqdm(data) :
        text_list = preprocess_data(json_data)
        text_data.extend(text_list)

    print('Tokenize Text Data')
    sen_data = []
    for text in tqdm(text_data) :
        sen_list = kss.split_sentences(text)
        sen_data.extend(sen_list)
        
    print('Size of Sentence Data : %d' %len(sen_data))

    print('Write Preprocessed Data')
    write_data(sen_data, text_path, preprocess_kor)

    print('Train Tokenizer')
    train_spm(args.dir, args.text, args.model, args.token_size)
    
if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./Data/Version1.0', help='Version1 Data Path')
    parser.add_argument('--text', type=str, default='kor_news.txt',  help='Text data file name')
    parser.add_argument('--model', type=str, default='kor_tokenizer',  help='Tokenizer file name')
    parser.add_argument('--dir', type=str, default='./Token',  help='File Writing Directory')
    parser.add_argument('--token_size', type=int, default=32000, help='Token Size (default: 32000)')
    args = parser.parse_args()
    train(args)
    
