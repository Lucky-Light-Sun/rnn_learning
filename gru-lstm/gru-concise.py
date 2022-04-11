import torch
import torch.nn as nn
import torch.nn.functional as F
import d2l.torch as d2l

import config
from tqdm import tqdm


class GRU(nn.Module):
    def __init__(self, num_hiddens, num_vocabs, num_layers=1):
        super(GRU, self).__init__()
        self.num_hiddens, self.num_vocabs = num_hiddens, num_vocabs
        num_inputs = num_outputs = num_vocabs
        self.gru = nn.GRU(input_size=num_inputs, hidden_size=num_hiddens,
                          num_layers=num_layers, bidirectional=False)
        self.linear = nn.Linear(num_hiddens, num_outputs)

    def weight_clip(self, theta):
        params = [p for p in self.parameters() if p.requires_grad]
        norm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in params))
        if norm > theta:
            for p in params:
                p.grad[:] *= theta / norm

    def init_state(self, batch_size):
        return (torch.zeros((1, batch_size, self.num_hiddens), device=config.DEVICE), )

    def forward(self, inputs, state):
        inputs = F.one_hot(inputs.T.long(), self.num_vocabs).to(torch.float32)
        H, = state
        X_tilda, H = self.gru(inputs, H)
        Y = self.linear(X_tilda.reshape(-1, X_tilda.shape[-1]))
        return Y, (H, )

d2l.RNNModel
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

        l = loss(Y_hat, Y.T.reshape(-1))
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
    net = GRU(config.NUM_HIDDENS, len(vocab)).to(config.DEVICE)

    # the trainer and loss
    trainer = torch.optim.SGD(net.parameters(), lr=config.LR)
    loss = nn.CrossEntropyLoss().to(config.DEVICE)

    for epoch in range(config.NUM_EPOCHS):
        train_epoch(epoch, data_iter, net, trainer, loss, use_random_iter=False)


def test():
    train()


if __name__ == '__main__':
    test()
