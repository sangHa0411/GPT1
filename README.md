# PretrainGPT

# Data Structure
```
|-- Data
├── dataset.py
├── loader.py
├── model.py
├── preprocessor.py
├── README.md
├── scheduler.py
├── Tokenizer
│   ├── kor_newspaper.txt
│   ├── tokenizer.model
│   └── tokenizer.vocab
├── tokenizer.py
├── train.py
└── train_tokenizer.py

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


# Model Specification
  1. GPT1 - Transformer Decoder
  2. Layer size : 12
  3. Embedding size : 768
  4. Hidden size : 3072
  5. Head size : 12
  6. Sequence size : 128

# Training 
  1. Epoch : 100
  2. Warmup staeps : 2000
  3. Optimizer : Adam
  4. Batch size : 128

# Source
  1. 모두의 말뭉치 : https://corpus.korean.go.kr
  2. 2020 신문 데이터
