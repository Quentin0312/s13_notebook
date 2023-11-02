from torchtext.vocab import vocab
from collections import Counter, OrderedDict
import spacy

# nlp = spacy.load("fr_core_news_md")
nlp = spacy.load("en_core_web_md")


# # tokens => string[]
# def create_vocab(tokens):
#     unk_token = "<unk>"
#     v2 = vocab(OrderedDict([(token, 1) for token in tokens]), specials=[unk_token])

#     # make default index same as index of unk_token
#     v2.set_default_index(v2[unk_token])

#     return v2


# tokens => string[]
def create_vocab(tokens):
    unk_token = "<unk>"
    counter = Counter(tokens)
    sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    ordered_dict = OrderedDict(sorted_by_freq_tuples)
    # Delete the last 25,000 items
    rangeValue = 25000 - len(ordered_dict)
    for _ in range(-rangeValue):
        ordered_dict.popitem()
    v2 = vocab(ordered_dict, specials=[unk_token])

    # make default index same as index of unk_token
    v2.set_default_index(v2[unk_token])

    return v2


def tokenize(text):
    return [token.text for token in nlp(text)]


def get_total_tokens(dataloader1):
    total_tokens = []
    for batch in dataloader1:
        tokens = tokenize(batch["text"][0][15:])
        for token in tokens:
            total_tokens.append(token)

    # for batch in dataloader2:
    #     tokens = tokenize(batch["text"][0][15:])
    #     for token in tokens:
    #         total_tokens.append(token)

    return total_tokens


def save_total_tokens(total_tokens):
    file = open("IMDB_total_tokens.txt", "w")
    for item in total_tokens:
        file.write(item + "\n")
    file.close()


def get_total_tokens_from_files():
    file = open("IMDB_total_tokens.txt", "r")
    total_tokens = file.read().split("\n")
    file.close()

    return total_tokens
