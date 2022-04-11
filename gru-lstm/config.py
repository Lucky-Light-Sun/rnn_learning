import torch


DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

THETA = 1
NUM_EPOCHS = 500
BATCH_SIZE = 32
NUM_HIDDENS = 256
NUM_STEPS = 35
LR = 1

