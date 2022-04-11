import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import config
from tqdm import tqdm


class RNN(nn.Module):
    def __init__(self, vocab_size, num_hiddens, batch_size):
        super(RNN, self).__init__()
        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens
        self.batch_size = batch_size
        num_inputs = num_outputs = vocab_size
        self.w_xh = nn.Parameter(self.normal((num_inputs, num_hiddens)))
        self.w_hh = nn.Parameter(self.normal((num_hiddens, num_hiddens)))
        self.w_ho = nn.Parameter(self.normal((num_hiddens, num_outputs)))
        self.b_h = nn.Parameter(torch.zeros(num_hiddens))
        self.b_o = nn.Parameter(torch.zeros(num_outputs))

    def normal(self, shape):
        return torch.randn(size=shape) * 0.01

    def init_status(self, batch_size):
        # return self._normal(self.batch_size, self.num_hiddens).to(self.w_hh.device)
        return (torch.zeros((batch_size, self.num_hiddens), device=self.w_hh.device), )

    def forward(self, X, state):
        X = F.one_hot(X.T, num_classes=self.vocab_size).type(torch.float32)
        Y = []
        H, = state
        for x in X:
            # 一定要注意书写激活函数
            H = torch.tanh(H @ self.w_hh + x @ self.w_xh + self.b_h)
            y = H @ self.w_ho + self.b_o
            Y.append(y.unsqueeze(0))
        # return torch.sigmoid(torch.cat(Y, dim=0)), status
        return torch.cat(Y, dim=0), (H, )  # Y: num_steps, batch_size, len(vocab)

    def grid_clip(self, theta):
        params = [p for p in self.parameters() if p.requires_grad]

        norm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in params))
        if norm > theta:
            for p in params:
                p.grad[:] *= theta / norm


def train_epoch(epoch, data_iter, net, loss, trainer, scaler: torch.cuda.amp.GradScaler, use_random=False):
    loop = tqdm(data_iter, leave=True, total=len(data_iter))
    state = None
    total_loss = []
    for idx, (x, y) in enumerate(loop):
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)

        if idx == 0 or use_random:
            state = net.init_status(x.shape[0])
        else:
            if isinstance(state, (tuple, list)):
                for min_state in state:
                    min_state.detach_()
            else:
                state.detach_()

        with torch.cuda.amp.autocast():
            y_hat, state = net(x, state)
            # Y: num_steps, batch_size, len(vocab)
            l = loss(y_hat.reshape(-1, y_hat.shape[-1]), y.T.reshape(-1).long()).mean()
            total_loss.append(l)

        # trainer.zero_grad()
        # scaler.scale(l).backward()
        # net.grid_clip(config.THETA)
        # scaler.step(trainer)
        # scaler.update()
        trainer.zero_grad()
        l.backward()
        net.grid_clip(config.THETA)
        trainer.step()

        loop.set_description(f'Epoch[{epoch+1}/{config.NUM_EPOCHS}]')
        loop.set_postfix(loss=l.item(), exp_loss=torch.exp(l).item())


def train():
    # the dataset
    batch_size, num_steps = config.BATCH_SIZE, config.NUM_STEPS
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)


    # the net
    net = RNN(len(vocab), config.NUM_HIDDENS, config.BATCH_SIZE).to(config.DEVICE)
    loss = nn.CrossEntropyLoss().to(config.DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    trainer = torch.optim.SGD(net.parameters(), lr=config.LR)

    for epoch in range(config.NUM_EPOCHS):
        train_epoch(epoch, train_iter, net, loss, trainer, scaler, use_random=False)


def predict(net, prefix):
    # 预热期计算 H
    # 验证期
    pass


def test():
    train()


if __name__ == '__main__':
    test()
