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

# Dependencies
  1. pandas : '1.1.4'
  2. numpy : '1.19.2' 
  3. torch : '1.9.0+cu102'
  4. konlpy : '0.5.2'

# Tokenizer Specification
  1. Subword Tokenizer : BPE
  2. Sentencepiece
  3. Vocab size : 35000
  4. Special Token
      * PAD : 0
      * UNK : 1
      * SOS : 2
      * EOS : 3
      * CLS : 4
      * SEP : 5

# Model Configuration
  1. BERT - Transformer Decoder Architecture
  2. Layer size : 12
  3. Embedding size : 768
  4. Hidden size : 3072
  5. Head size : 12
  6. Sequence size : 256
  7. DropOut Rate : 1e-1
  8. LayerNormalization : 1e-6

# Training Configuration
  1. Epoch : 30
  2. Warmup staps : 2000
  3. Optimizer : Adam
        1. Betas : (0.9, 0.98)
        2. Weight Decay : 1e-4
  4. Batch size : 64

# Source
  1. 모두의 말뭉치 : https://corpus.korean.go.kr
  2. 2020 신문 데이터
