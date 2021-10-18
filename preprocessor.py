import re

class SenPreprocessor :
    def __init__(self, tokenizer) :
        assert hasattr(tokenizer, 'morphs')
        self.tokenizer = tokenizer

    def __call__(self, sen) :
        assert isinstance(sen, str)
        try :
            tok_list = self.tokenizer.morphs(sen)
            sen = ' '.join(tok_list)
            return sen
        except ValueError as e :
            return None
 