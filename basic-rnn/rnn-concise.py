import torch
import torch.nn as nn
import torch.nn.functional as F
import d2l.torch as d2l

import config
from tqdm import tqdm


class RNN(nn.Module):
    def __init__(self, num_vocab, num_hiddens, num_layers):
        super(RNN, self).__init__()
        self.num_vocab = num_vocab
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size=num_vocab, hidden_size=num_hiddens, num_layers=num_layers)
        self.linear = nn.Linear(in_features=num_hiddens, out_features=num_vocab)

    def init_state(self, batch_size):
        return torch.zeros((self.num_layers, batch_size, self.num_hiddens), device=config.DEVICE)

    def weight_clipping(self, theta):
        params = [p for p in self.parameters() if p.requires_grad]

        norm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in params))
        if norm > theta:
            for p in params:
                p.grad[:] *= theta / norm

    def forward(self, inputs, state):
        inputs = F.one_hot(inputs.T.long(), self.num_vocab).to(torch.float32)
        outputs, state = self.rnn(inputs, state)
        outputs = self.linear(outputs.reshape(-1, self.num_hiddens))  # outputs 这个可不行乱T。。。绝了
        return outputs, state


def train_epoch(epoch, data_iter, net, loss, trainer, random=False):
    loop = tqdm(data_iter, total=len(data_iter), leave=True)
    for idx, (x, y) in enumerate(loop):
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)
        if idx == 0 or random is True:
            state = net.init_state(x.shape[0])
        else:
            if isinstance(state, (tuple, list)):
                for min_state in state:
                    min_state.detach_()
            else:
                state.detach_()
        y_hat, state = net(x, state)
        l = loss(y_hat, y.T.reshape(-1))    # 在forward函数中已经T完了，这里y不需要进行T了

        trainer.zero_grad()
        l.backward()
        net.weight_clipping(config.THETA)
        trainer.step()

        loop.set_description(f'Epoch[{epoch+1}/{config.NUM_EPOCHS}]')
        loop.set_postfix(loss=l.item(), exp_loss=torch.exp(l).item())


def train():
    # load the dataset
    data_iter, vocab = d2l.load_data_time_machine(config.BATCH_SIZE, config.NUM_STEPS, use_random_iter=False)

    # the model
    net = RNN(len(vocab), config.NUM_HIDDENS, config.NUM_LAYERS).to(config.DEVICE)

    # the loss
    loss = nn.CrossEntropyLoss().to(config.DEVICE)
    trainer = torch.optim.SGD(net.parameters(), config.LR)

    for epoch in range(config.NUM_EPOCHS):
        train_epoch(epoch, data_iter, net, loss, trainer)
    print('let\'s do the predict')
    predict('the time machine', net, vocab, 50)


def predict(prefix, net: RNN, vocab, predict_len):
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor(outputs[-1], device=config.DEVICE).reshape(1, 1)
    state = net.init_state(1)

    # 预热
    for ch in prefix[1:]:
        y, state = net(get_input(), state)
        outputs.append(vocab[ch])

    # 预测
    for i in range(predict_len):
        y, state = net(get_input(), state)
        outputs.append(int(y.reshape(-1).argmax(dim=0)))
    print(''.join(vocab.to_tokens(outputs)))


if __name__ == '__main__':
    train()
