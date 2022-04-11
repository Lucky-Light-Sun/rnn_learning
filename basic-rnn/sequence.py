import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def test():
    T = 1000
    x = torch.arange(0, T, 1, dtype=torch.float)
    y = torch.sin(x * 0.01) + torch.normal(0, 0.2, (T, ))
    # d2l.plot(x, [y], 'x', 'y', xlim=[0, 1000], figsize=(6, 3))
    axis = plt.gca()
    axis.plot(x, y)
    axis.set_xlim([0, T])
    axis.grid()
    axis.set_title('y = sin(0.01x) + normal(0, 0.2)')
    axis.set_xlabel('x')
    axis.set_ylabel('y')
    plt.show()


def get_data_iter(T=1000, tau=4):

    time = torch.arange(0, T, 1, dtype=torch.float)
    x = torch.sin(time * 0.01) + torch.normal(0, 0.2, (T,))

    # 注意是 T - tau，而不是 T - tau + 1， 因为后面还有一个特征
    features = torch.zeros((T - tau, tau))
    labels = torch.zeros((T - tau, ))
    for i in range(T - tau):
        features[i][:] = x[i:i+tau]
        labels[i] = x[i + tau]

    # 也可以这样初始化，更快一些
    for i in range(tau):
        features[:, i] = x[i:i+T-tau]   # 区分features[:][i] = x[i:i+T-tau]
    labels = x[tau:].reshape(-1, 1)     # 最好 reshape一下子
    # print(features[:10])
    # print(labels[:10])

    batch_size, n_train = 16, 600
    dataset = TensorDataset(features[:n_train], labels[:n_train])
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2), time, features, labels


def get_net(input, output):
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight.data)

    net = nn.Sequential(
        nn.Linear(input, 10),
        nn.ReLU(),
        nn.Linear(10, output)
    )
    net.apply(weight_init)
    return net


def train():
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    T = 1000
    tau = 4
    lr = 0.01
    num_epochs = 5

    net = get_net(tau, 1).to(device)
    loss = nn.MSELoss().to(device)
    data_iter, time, features, labels = get_data_iter()

    trainer = torch.optim.Adam(net.parameters(), lr)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):
        loop = tqdm(data_iter, leave=True)
        for i, (X, y) in enumerate(loop):
            with torch.cuda.amp.autocast():
                X, y = X.to(device), y.to(device)
                y_hat = net(X)
                l = loss(y_hat, y)

            trainer.zero_grad()
            scaler.scale(l).backward()
            scaler.step(trainer)
            scaler.update()
            loop.set_description(f'Epoch[{epoch+1}/{num_epochs}]')
            loop.set_postfix(loss=l.detach().item())

    with torch.no_grad():
        onestep_predicts = net(features.to(device))
        d2l.plot([time[tau:], time[tau:]], [labels.detach().cpu(), onestep_predicts.detach().cpu()], 'time', 'x',
                 legend=['data', 'one step'], xlim=[0, T], figsize=(6, 3))
        plt.show()


if __name__ == '__main__':
    train()
