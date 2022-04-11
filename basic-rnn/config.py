import torch


DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
NUM_HIDDENS = 512
BATCH_SIZE = 32
NUM_STEPS = 35
NUM_EPOCHS = 500
NUM_LAYERS = 1
LR = 1
THETA = 1

