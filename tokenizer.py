
import os
import sentencepiece as spm
from dataset import Token

spm_templates= '--input={} \
--pad_id={} \
--bos_id={} \
--eos_id={} \
--unk_id={} \
--user_defined_symbols={} \
--model_prefix={} \
--vocab_size={} \
--character_coverage={} \
--model_type={}'

def write_data(text_list, text_path) :
    with open(text_path, 'w') as f :
        for sen in text_list :
            f.write(sen + '\n')

def train_spm(text_path, model_path, vocab_size) :
    spm_cmd = spm_templates.format(text_path, 
        Token.PAD,
        Token.SOS, 
        Token.EOS, 
        Token.UNK, 
        model_path, 
        vocab_size, 
        1.0, 
        'bpe')
    spm.SentencePieceTrainer.Train(spm_cmd)

def get_spm(model_path) :
    if os.path.exists(model_path) :
        sp = spm.SentencePieceProcessor()
        sp.Load(model_path)
        sp.SetEncodeExtraOptions('bos:eos')
        return sp
    else:
        raise FileNotFoundError