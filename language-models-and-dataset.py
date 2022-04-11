import torch
import d2l.torch as d2l
import random
import matplotlib.pyplot as plt


def test2():
    text = d2l.read_time_machine()
    tokens = d2l.tokenize(text)
    corpus = [token for line in tokens for token in line]
    vocab = d2l.Vocab(tokens)

    data_iter = seq_data_iter_sequential(vocab[corpus], 2, 4)
    for X, y in data_iter:
        print(X)
        print(y)


def test():
    text = d2l.read_time_machine()
    tokens = d2l.tokenize(text)
    corpus = [token for line in tokens for token in line]


    vocab = d2l.Vocab(tokens)
    freqs = [freq for token, freq in vocab.token_freqs]

    bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
    bigram_vocab = d2l.Vocab(bigram_tokens)
    bi_freqs = [freq for token, freq in bigram_vocab.token_freqs]

    trigram_tokens = [triple for triple in zip(corpus[:-2], corpus[1:-1], corpus[2:])]
    trigram_vocab = d2l.Vocab(trigram_tokens)
    tri_freqs = [freq for token, freq in trigram_vocab.token_freqs]

    axis = plt.gca()
    axis.plot(list(range(len(freqs))), freqs)
    axis.plot(list(range(len(bi_freqs))), bi_freqs)
    axis.plot(list(range(len(tri_freqs))), tri_freqs)

    axis.set_xlabel('token x')
    axis.set_ylabel('frequency: n(x)')
    axis.set_xscale('log')
    axis.set_yscale('log')
    axis.grid()
    axis.legend(['uni_freqs', 'bi_freqs', 'tri_freqs'])
    plt.show()


def seq_data_iter_random(corpus, batch_size, num_steps):
    start = random.randint(0, num_steps-1)
    corpus = corpus[start:]
    num_batch = (len(corpus) - 1) // num_steps // batch_size
    start_idx = list(range(0, num_batch * batch_size * num_steps, num_steps))
    random.shuffle(start_idx)
    for batch in range(num_batch):
        x_list = [corpus[start_idx[i+batch*batch_size]: start_idx[i+batch*batch_size]+num_steps]
                  for i in range(batch_size)]
        y_list = [corpus[start_idx[i+batch*batch_size]+1: start_idx[i+batch*batch_size] + num_steps+1]
                  for i in range(batch_size)]
        yield torch.tensor(x_list), torch.tensor(y_list)


def seq_data_iter_sequential(corpus, batch_size, num_steps):
    corpus = corpus[random.randint(0, num_steps - 1):]
    num_batches = (len(corpus) - 1) // num_steps // batch_size

    x = torch.tensor(corpus[:num_batches * num_steps * batch_size]).reshape(batch_size, -1)
    y = torch.tensor(corpus[1:num_batches * num_steps * batch_size+1]).reshape(batch_size, -1)
    for batch in range(num_batches):
        yield x[:, batch*num_steps:batch*num_steps+num_steps], y[:, batch*num_steps:batch*num_steps+num_steps]


class SeqDataLoader:
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens=-1):
        self.batch_size, self.num_steps = batch_size, num_steps
        self.corpus, self.vocab = d2l.load_corpus_time_machine(max_tokens)
        if use_random_iter:
            self.iter_fn = d2l.seq_data_iter_random
        else:
            self.iter_fn = d2l.seq_data_iter_sequential

    def __iter__(self):
        return self.iter_fn(corpus=self.corpus, batch_size=self.batch_size, num_steps=self.num_steps)


def load_time_machine(batch_size, num_steps, use_random_iter=False, max_tokens=10000):
    data_iter = SeqDataLoader(batch_size=batch_size, num_steps=num_steps, use_random_iter=use_random_iter, max_tokens=max_tokens)
    return data_iter, data_iter.vocab


if __name__ == '__main__':
    test2()
