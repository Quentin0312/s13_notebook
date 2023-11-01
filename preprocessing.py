from torchtext.vocab import vocab
from collections import OrderedDict


# tokens => string[]
def create_vocab(tokens):
    unk_token = "<unk>"
    v2 = vocab(OrderedDict([(token, 1) for token in tokens]), specials=[unk_token])

    # make default index same as index of unk_token
    v2.set_default_index(v2[unk_token])

    return v2
