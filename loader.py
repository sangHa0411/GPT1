import os
import json
import random
from tqdm import tqdm

def read_data(file_path) :
    with open(file_path, 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)
    return json_data

def get_data(dir_path, size) :
    data_list = []
    file_list = os.listdir(dir_path)
    assert size < len(file_list)
    for file in tqdm(file_list[:size]) :
        if file.endswith('.json') :
            try :
                file_path = os.path.join(dir_path, file)
                data = read_data(file_path)
                data_list.append(data)
            except UnicodeDecodeError :
                continue
            except json.JSONDecodeError :
                continue 
    return data_list

def preprocess_data(json_data) :
    doc_list = [doc['paragraph'] for doc in json_data['document']]
    text_data = []
    for doc in doc_list :
        text_list = [text['form'] for text in doc] 
        text_data += text_list
    return text_data

