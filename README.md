# PretrainGPT

# Data Structure
```
.
├── Log
├── README.md
├── Tokenizer
│   ├── make_tokenizer.py
│   └── tokenizer.py
├── __pycache__
├── dataset.py
├── loader.py
├── model.py
├── preprocessor.py
├── scheduler.py
└── train.py
```

# Tokenizer Specification
  1. Subword Tokenizer
  2. Sentencepiece
  3. Vocab size : 32000
  4. Special Token
      * PAD : 0
      * UNK : 1
      * SOS : 2
      * EOS : 3
      * CLS : 4
      * SEP : 5


# Model Specification
  1. GPT1 - Transformer Decoder
  2. Layer size : 12
  3. Embedding size : 768
  4. Hidden size : 3072
  5. Head size : 12
  6. Sequence size : 256

# Training 
  1. Epoch : 100
  2. Warmup staps : 2000
  3. Optimizer : Adam
  4. Batch size : 64

# Source
  1. 모두의 말뭉치 : https://corpus.korean.go.kr
  2. 2020 신문 데이터
