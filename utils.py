import numpy as np

def tokenize(words, vocabulary):
    tokens = []
    vocab_size = len(vocabulary)
    token_map = [[w, i] for w, i in zip(vocabulary, range(vocab_size))]
    token_map = dict(token_map)
    for w in words:
        if w.lower() in token_map.keys():
            tokens.append(token_map[w.lower()])
        else:
            #   print(w, 'not in vocab')
            tokens.append(vocab_size)
    return tokens

def mytokenizer(text, vocab_size=10000):
    def replacetokens(tokens):
        pairs = {}
        maxocc = 2
        maxpair = ''
        for i in range(len(tokens)-1):
            pair = tokens[i]+tokens[i+1]
            if pair in pairs:
                pairs[pair] += 1
            else:
                pairs[pair] = 1
            if pairs[pair] > maxocc:
                maxocc = pairs[pair]
                maxpair = pair
        if maxpair == '':
            return tokens
        print('"'+maxpair+'"')
        res = []
        for i in range(len(tokens)-1):
            if tokens[i]+tokens[i+1] == maxpair:
                res.append(maxpair)
            elif i>0 and tokens[i-1]+tokens[i] == maxpair:
                continue
            else:
                res.append(tokens[i])
        return res
    tokens = list(text)
    while len(np.unique(tokens))<vocab_size:
        prev_len = len(tokens)
        tokens = replacetokens(tokens)
        if len(tokens) == prev_len:
            return tokens
    return tokens


def other_tokenizer(text):
    from collections import Counter
    # Initial tokenization
    tokens = list(text)
    pairs = Counter(zip(tokens[:-1], tokens[1:]))
    while pairs:
        # Find the most common pair
        most_common = pairs.most_common(1)[0][0]
        # Merge the pair
        new_token = ''.join(most_common)
        tokens = [new_token if (tokens[i] == most_common[0] and tokens[i + 1] == most_common[1]) else tokens[i] for i in range(len(tokens) - 1)]
        pairs = Counter(zip(tokens[:-1], tokens[1:]))
    return tokens


def fast_tokenizer(text):
    from fast_bpe_tokenizer import Tokenizer
    tokenizer = Tokenizer(text)
    return tokenizer.encode(text)

def text_to_words(text, delim):
    for d in delim:
        text = f' {d} '.join(text.split(d))
    return text.split()


def split_upper(text, d, maj=False):
    split = text.split(f' {d} ')
    if maj:
        for i, w in enumerate(split):
            if len(w) > 0:
                split[i] = w[0].upper() + w[1:]
    return split


def words_to_text(words, d0, dL, dR, dRL, dMaj):
    text = ' '.join(words)
    for d in d0:
        split = split_upper(text, d, maj=d in dMaj)
        text = f'{d}'.join(split)
    for d in dL:
        split = split_upper(text, d, maj=d in dMaj)
        text = f' {d}'.join(split)
    for d in dR:
        split = split_upper(text, d, maj=d in dMaj)
        text = f'{d} '.join(split)
    for d in dRL:
        split = split_upper(text, d, maj=d in dMaj)
        text = f' {d} '.join(split)
    text = text.replace('newline', '\n')
    text = text.replace('Newline', '\n')
    return text





def one_hot(tokens, vocab_size):
    ohmap = np.zeros((vocab_size, vocab_size))
    ohmap[np.arange(len(ohmap)), np.arange(len(ohmap))] = 1
    onehot = []
    for t in tokens:
        onehot.append(ohmap[t])
    return onehot


def remove_unknown(tokens, t_size, vocab_size):
    tokens = np.array(tokens)
    rollmax = np.array(pd.Series(tokens).rolling(t_size).max().shift(-t_size + 1))
    return tokens[rollmax < vocab_size]


def train_val_split(inputs, split=.2):
    train_lines = []
    val_lines = []
    for text_input in inputs:
        if 'WikiQA' in text_input:
            encoding = 'utf-8'
        else:
            encoding = 'unicode_escape'
        with open(text_input, encoding=encoding) as f:
            lines = f.readlines()
        train_lines += lines[int(split * len(lines)):] + [' newbook ']
        val_lines += lines[:int(split * len(lines))] + [' newbook ']
    train_text = ' newline '.join(train_lines)
    val_text = ' newline '.join(val_lines)
    return train_text, val_text


def words_from_texts(inputs, split=.2): 
    train_text, val_text = train_val_split(inputs, split)
    train_w = text_to_words(train_text, all_delims)
    val_w = text_to_words(val_text, all_delims)
    vocabulary, counts = np.unique([w.lower() for w in train_w + val_w], return_counts=True)
    vocabulary = vocabulary[counts > min_token_count * len(train_w + val_w)]
    print('Vocabulary size: ', len(vocabulary))
    print('Train:', len(train_w))
    print('Val:', len(val_w))
    return train_w, val_w, vocabulary


def tokens_from_words(words, vocabulary):
    tokens = tokenize(words, vocabulary)
    print(len(tokens))
    tokens = remove_unknown(tokens, block_size, len(vocabulary))
    print(len(tokens))
    return tokens


def get_batch(b_idx, t_size, tokens):
    x = []
    y = []
    for i in b_idx:
        x.append(tokens[i:i + t_size])
        y.append(one_hot(tokens[i + 1:i + t_size + 1], len(vocab)))
    return np.array(x), np.array(y)