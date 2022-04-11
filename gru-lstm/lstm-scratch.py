import torch
import torch.nn as nn
import torch.nn.functional as F
import d2l.torch as d2l

import config
from tqdm import tqdm


class LSTM(nn.Module):
    def __init__(self, num_hiddens, num_vocabs):
        super(LSTM, self).__init__()
        self.num_hiddens, self.num_vocabs = num_hiddens, num_vocabs
        num_inputs = num_outputs = num_vocabs
        self.f_w_xh, self.f_w_hh, self.f_b = self._normalize(num_inputs, num_hiddens)
        self.i_w_xh, self.i_w_hh, self.i_b = self._normalize(num_inputs, num_hiddens)
        self.c_w_xh, self.c_w_hh, self.c_b = self._normalize(num_inputs, num_hiddens)
        self.o_w_xh, self.o_w_hh, self.o_b = self._normalize(num_inputs, num_hiddens)
        self.y_w_ho = nn.Parameter(torch.randn((num_hiddens, num_outputs)) * 0.01)
        self.y_b_o = nn.Parameter(torch.zeros(num_outputs))

    def _normalize(self, num_inputs, num_hiddens):
        return (
            nn.Parameter(torch.randn((num_inputs, num_hiddens)) * 0.01),
            nn.Parameter(torch.randn((num_hiddens, num_hiddens)) * 0.01),
            nn.Parameter(torch.zeros(num_hiddens)),
        )

    def weight_clip(self, theta):
        params = [p for p in self.parameters() if p.requires_grad]

        norm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in params))
        if norm > theta:
            for p in params:
                p.grad[:] *= theta / norm

    def init_state(self, batch_size):
        return (torch.zeros((batch_size, self.num_hiddens), device=config.DEVICE),
                torch.zeros((batch_size, self.num_hiddens), device=config.DEVICE))

    def forward(self, inputs, state):
        inputs = F.one_hot(inputs.T.long(), self.num_vocabs).to(torch.float32)
        outputs = []
        H, C = state
        for X in inputs:
            f = torch.sigmoid(X @ self.f_w_xh + H @ self.f_w_hh + self.f_b)
            i = torch.sigmoid(X @ self.i_w_xh + H @ self.i_w_hh + self.i_b)
            o = torch.sigmoid(X @ self.o_w_xh + H @ self.o_w_hh + self.o_b)
            C_tilta = torch.tanh(X @ self.c_w_xh + H @ self.c_w_hh + self.c_b)
            C = f * C + i * C_tilta
            H = o * C
            Y = H @ self.y_w_ho + self.y_b_o
            outputs.append(Y.unsqueeze(0))
        return torch.cat(outputs, dim=0), (H, C)


def train_epoch(epoch, data_iter, net, trainer, loss, use_random_iter=False):
    loop = tqdm(data_iter, leave=True, total=len(data_iter))
    state = None
    for idx, (X, Y) in enumerate(loop):
        X, Y = X.to(config.DEVICE), Y.to(config.DEVICE)
        if idx == 0 or use_random_iter:
            state = net.init_state(X.shape[0])
        else:
            if isinstance(state, (list, tuple)):
                for s in state:
                    s.detach_()
            else:
                state.detach_()
        Y_hat, state = net(X, state)

        l = loss(Y_hat.reshape(-1, Y_hat.shape[-1]), Y.T.reshape(-1))
        trainer.zero_grad()
        l.backward()
        net.weight_clip(config.THETA)
        trainer.step()

        loop.set_description(f'Epoch[{epoch+1}/{config.NUM_EPOCHS}]')
        loop.set_postfix(l=l.item(), exp_l=torch.exp(l).item())


def train():
    # the dataset
    data_iter, vocab = d2l.load_data_time_machine(config.BATCH_SIZE, config.NUM_STEPS, use_random_iter=False)

    # the model
    net = LSTM(config.NUM_HIDDENS, len(vocab)).to(config.DEVICE)
    # inputs = torch.randint(0, len(vocab) - 1, (config.BATCH_SIZE, config.NUM_STEPS), device=config.DEVICE)
    # state = net.init_state(config.BATCH_SIZE)
    # net(inputs, state)

    # the trainer and loss
    trainer = torch.optim.SGD(net.parameters(), lr=config.LR)
    loss = nn.CrossEntropyLoss().to(config.DEVICE)

    for epoch in range(config.NUM_EPOCHS):
        train_epoch(epoch, data_iter, net, trainer, loss, use_random_iter=False)


def test():
    train()


if __name__ == '__main__':
    test()
