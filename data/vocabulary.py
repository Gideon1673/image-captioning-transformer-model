"""
File này phụ trách:
Làm sạch caption -> Tokenize -> Xây vocabulary -> Text → token IDs -> Token IDs → text
"""


class Vocabulary:
    PAD_TOKEN = "<pad>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"

    def __init__(self, min_frequency=5):
        self.word_to_index = {}
        self.index_to_word = {}
        self.min_frequency = min_frequency

    def tokenize(self, text):
        pass

    def build(self, captions):
        pass

    def encode(self, caption):
        pass

    def decode(self, token_ids):
        pass

    def save(self, path):
        pass

    @classmethod
    def load(cls, path):
        pass
