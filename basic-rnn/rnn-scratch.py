import d2l.torch as d2l

import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from tqdm import tqdm


class RNN(nn.Module):
    def __init__(self, vocab_size, num_hiddens, batch_size):
        super(RNN, self).__init__()
        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens
        self.batch_size = batch_size
        input = output = vocab_size
        self.w_xh = nn.Parameter(self._normal(input, num_hiddens))
        self.w_hh = nn.Parameter(self._normal(num_hiddens, num_hiddens))
        self.w_ho = nn.Parameter(self._normal(num_hiddens, output))
        self.b_h = nn.Parameter(torch.zeros((num_hiddens,)))
        self.b_o = nn.Parameter(torch.zeros((output,)))

    def _normal(self, h, w):
        return torch.normal(0, 0.01, (h, w))

    def init_status(self, batch_size):
        # return self._normal(self.batch_size, self.num_hiddens).to(self.w_hh.device)
        return torch.zeros((batch_size, self.num_hiddens), device=self.w_hh.device)

    def forward(self, X, status):
        X = F.one_hot(X.T, num_classes=self.vocab_size).type(torch.float32)
        Y = []
        for x in X:     # 有趣的是我们并没有明确规定 t 是多少
            # 一定要注意书写激活函数
            status = torch.tanh(status @ self.w_hh + x @ self.w_xh + self.b_h)
            y = status @ self.w_ho + self.b_o
            # Y.append(torch.softmax(y.unsqueeze(0), dim=-1))
            Y.append(y.unsqueeze(0))
        # return torch.sigmoid(torch.cat(Y, dim=0)), status
        return torch.cat(Y, dim=0), status  # Y: num_steps, batch_size, len(vocab)

    def grid_clip(self, theta):
        params = [p for p in self.parameters() if p.requires_grad]

        norm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in params))
        if norm > theta:
            for p in params:
                p.grad[:] *= theta / norm


def train_epoch(epoch, data_iter, net, loss, trainer, scaler: torch.cuda.amp.GradScaler, use_random=False):
    loop = tqdm(data_iter, leave=True, total=len(data_iter))
    for idx, (x, y) in enumerate(loop):
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)

        if idx == 0 or use_random:
            status = net.init_status(config.BATCH_SIZE)
        else:
            status.detach_()

        with torch.cuda.amp.autocast():
            y_hat, status = net(x, status)
            # Y: num_steps, batch_size, len(vocab)
            l = loss(y_hat.reshape(-1, y_hat.shape[-1]), y.T.reshape(-1))

        # trainer.zero_grad()
        # scaler.scale(l).backward()
        # # net.grid_clip(config.THETA)   主要是 grid_clip 这一部分操作和 autocast 进行冲突了！！
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
    data_iter, vocab = d2l.load_data_time_machine(config.BATCH_SIZE, config.NUM_STEPS, use_random_iter=False)

    # the net
    net = RNN(len(vocab), config.NUM_HIDDENS, config.BATCH_SIZE).to(config.DEVICE)
    loss = nn.CrossEntropyLoss().to(config.DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    trainer = torch.optim.SGD(net.parameters(), lr=config.LR)

    for epoch in range(config.NUM_EPOCHS):
        train_epoch(epoch, data_iter, net, loss, trainer, scaler, use_random=False)
    print('this is the prediction')
    predict(net, 'the time machine', vocab, 50)


def predict(net: RNN, prefix, vocab, predict_len):
    state = net.init_status(batch_size=1)
    output = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor(output[-1], device=config.DEVICE).reshape(1, 1)
    # 预热期计算 H
    for ch in prefix[1:]:
        y, state = net(get_input(), state)    # 这个 y 根本用不到
        output.append(vocab[ch])
    # 验证期
    for i in range(predict_len):
        y, state = net(get_input(), state)
        # 使用 item() 或者是使用 int 强制类型转换是为了防止 get_input 中 torch.tensor 老是给警报
        output.append(y.reshape(-1).argmax().item())
    print(''.join(vocab.to_tokens(output)))


def test():
    train()


if __name__ == '__main__':
    test()
