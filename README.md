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

# Model Specification
  1. Layer size : 6
  2. Embedding size : 512
  3. Hidden size : 2048
  4. Head size : 8
  5. Sequence size : 12

# Training 
  1. Epoch : 100
  2. Warmup staeps : 2000
  3. Optimizer : Adam
  4. Batch size : 128

# Source
  1. 모두의 말뭉치 : https://corpus.korean.go.kr
  2. 2020 신문 데이터
