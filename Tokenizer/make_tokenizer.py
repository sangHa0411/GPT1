import os
import re
import sys
import argparse
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from konlpy.tag import Mecab

def make(args) :
    sys.path.append('../')
    from tokenizer import write_data, train_spm
    from loader import get_data, preprocess_data
    from preprocessor import SenPreprocessor

    print('Get Newspaper Data')
    data = get_data(args.data_dir, args.file_size)

    print('Extract Text Data')
    text_data = []
    for json_data in tqdm(data) :
        text_list = preprocess_data(json_data)
        text_data.extend(text_list)

    print('Tokenizing Data')
    sen_data = []
    for text in tqdm(text_data) :
        sen_list = sent_tokenize(text)
        sen_data.extend(sen_list)
        
    print('Size of Sentence Data : %d \n' %len(sen_data))

    print('Preprocessing Data')
    mecab = Mecab()
    sen_preprocessor = SenPreprocessor(mecab)
    sen_preprocessed = []
    for sen in tqdm(sen_data) :
        if len(sen) > args.max_size :
            continue
        sen = sen_preprocessor(sen)
        if sen != None :
            sen_preprocessed.append(sen)

    print('Write Text Data')
    text_path = os.path.join(args.tokenizer_dir, 'kor_newspaper.txt')
    write_data(sen_preprocessed, text_path)

    print('Train Tokenizer')
    train_spm(text_path, os.path.join(args.tokenizer_dir, 'tokenizer'), args.token_size)

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../../Data', help='Korean Newspaper Data directory')
    parser.add_argument('--max_size', type=int, default=256, help='max length of sentence')
    parser.add_argument('--file_size', type=int, default=10, help='size of newspaper file')
    parser.add_argument('--tokenizer_dir', type=str, default='./',  help='File Writing Directory')
    parser.add_argument('--token_size', type=int, default=35000, help='Token Size (default: 35000)')
    args = parser.parse_args()

    make(args)
    
