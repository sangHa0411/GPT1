import os
import sys
import json
import argparse
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from preprocessor import *

def read(file_path) :
    with open(file_path, 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)
    return json_data

def split_data(json_data) :
    doc_list = [doc['paragraph'] for doc in json_data['document']]
    sen_data = []
    for doc in doc_list :
        text_list = [text['form'] for text in doc] 
        sen_list = [sent_tokenize(text) for text in text_list]
        sen_list = sum(sen_list, [])
        sen_data += sen_list
    return sen_data

def get_data(dir_path) :
    data_list = []
    file_list = os.listdir(dir_path)
    for file in tqdm(file_list) :
        if file.endswith('.json') :
            try :
                file_path = os.path.join(dir_path, file)
                data = read(file_path)
                data_list.append(data)
            except UnicodeDecodeError :
                continue
            except json.JSONDecodeError :
                continue 
    return data_list

def train(args) :
    print('Get Newspaper Data')
    data = get_data(args.data)
    
    text_path = os.path.join(args.dir, args.text)
    print('Preprocess Raw Data')
    text_data = []
    for json_data in tqdm(data) :
        text_list = split_data(json_data)
        text_data.extend(text_list)
    
    min_size = args.min_sen_size
    max_size = args.max_sen_size
    print('Filtering Data by length')
    text_data = [text for text in text_data if (len(text) >= min_size and len(text) <= max_size)]

    print('Write Preprocessed Data')
    write_data(text_data, text_path, preprocess_kor)

    print('Train Tokenizer')
    train_spm(args.dir, args.text, args.model, args.token_size)
    

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./Data/Version1.0', help='Version1 Data Path')
    parser.add_argument('--text', type=str, default='kor_data.txt',  help='Text data file name')
    parser.add_argument('--model', type=str, default='kor_tokenizer',  help='Tokenizer file name')
    parser.add_argument('--dir', type=str, default='./Token',  help='File Writing Directory')
    parser.add_argument('--max_sen_size', type=int, default=300, help='Maximum of setnence size (default: 300)')
    parser.add_argument('--min_sen_size', type=int, default=10, help='Minimum of sentence size (default: 10)')
    parser.add_argument('--token_size', type=int, default=32000, help='Token Size (default: 32000)')
    args = parser.parse_args()
    train(args)
    

