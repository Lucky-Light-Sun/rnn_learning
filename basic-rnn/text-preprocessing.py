import collections
import re
from d2l import torch as d2l
import torch


# 读取文件
def read_time_machine(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    # + 表示一个或者多个，然后用 一个空格替换
    ret = [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]
    # print(lines[0])
    # print(ret[0])
    return ret


# lines 是字符串的列表
def tokenise(lines: str, token='word'):
    if token == 'word':
        return [line.split(' ') for line in lines if line != '']
    elif token == 'char':
        return [list(line) for line in lines if line != '']
    else:
        print('错误，未知词元类型')


class Vocab:
    """
    下标变词元，词元变下标。并且加入 词频统计， freq，保留词，unk的过滤条件
    """
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        counter = count_corpus(tokens)
        self.freq = sorted(counter.items(), key=lambda val: val[1], reverse=True)

        self.unk, uniq_tokens = 0, ['<unk>'] + reserved_tokens
        uniq_tokens += [token for token, freq in self.freq if token not in uniq_tokens and freq >= min_freq]

        self.idx_token, self.token_idx = [], dict()
        for idx, token in enumerate(uniq_tokens):
            self.idx_token.append(token)
            self.token_idx[token] = idx

    def __len__(self):
        return len(self.idx_token)

    def __getitem__(self, tokens):
        if isinstance(tokens, (tuple, list)):
            return [self.__getitem__(token) for token in tokens]
        else:
            return self.token_idx[tokens]

    def to_tokens(self, idx):
        if isinstance(idx, (tuple, list)):
            return [self.token_idx(i) for i in idx]
        else:
            return self.idx_token[idx]


def count_corpus(tokens=None):
    """
    :param tokens: 词元 tokens，可以是列表的列表，也可以是一个含有词元的一维列表
    :return: 返回的是统计完数量的词元
    """
    if len(tokens) == 0 or isinstance(tokens, list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


def test(filename):
    file = read_time_machine(filename)
    tokens = tokenise(file, token='word')
    vocab = Vocab(tokens=tokens)
    print(vocab.freq[:10])
    print(list(vocab.token_idx.items())[:10])

    for i in range(3):
        print(tokens[i])
        print(vocab[tokens[i]])


def load_corpus_time_machine(filename='../data/timemachine.txt', max_tokens=-1):
    lines = read_time_machine(filename)
    tokens = tokenise(lines, token='char')
    vocab = Vocab(tokens, min_freq=0, reserved_tokens=None)

    corpus = [vocab[token] for line in tokens for token in line]

    if max_tokens >= 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab


if __name__ == '__main__':
    filename = r'../../data/timemachine.txt'
    corpus, vocab = load_corpus_time_machine(filename)
    print(len(corpus))
    print(len(vocab))