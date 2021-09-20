# PretrainGPT

# Data Structure
```
|-- Data
|-- Log
|-- Model
|-- Token
|   |-- kor_data.txt
|   |-- kor_tokenizer.model
|   `-- kor_tokenizer.vocab
|-- dataset.py
|-- loader.py
|-- model.py
|-- preprocessor.py
|-- train.py
`-- train_tokenizer.py
```

# Tokenizer Specification
  1. Subword Tokenizer
  2. Sentencepiece
  3. Vocab size : 24000
  4. Special Token
      * PAD : 0
      * UNK : 1
      * SOS : 2
      * EOS : 3
      * SEP : 24000


# Model Specification
  1. GPT1 - Transformer Decoder
  2. Layer size : 6
  3. Embedding size : 512
  4. Hidden size : 2048
  5. Head size : 8
  6. Sequence size : 12

# Training 
  1. Epoch : 100
  2. Warmup staeps : 2000
  3. Optimizer : Adam
  4. Batch size : 128

# Source
  1. 모두의 말뭉치 : https://corpus.korean.go.kr
  2. 2020 신문 데이터
