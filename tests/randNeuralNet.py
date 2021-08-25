INT_LIMIT = 1

import torch
from torch import nn

class RandPwlNeuralNet(nn.Module):

    def __init__(self, inputDim, hiddenDim, hiddenNum):
        super(RandPwlNeuralNet, self).__init__()

        hl = []

        hl.append(nn.Linear(inputDim, hiddenDim))
        hl[-1].weight.data = torch.rand(hiddenDim, inputDim) + torch.randint(-INT_LIMIT, INT_LIMIT, (hiddenDim, inputDim))
        hl[-1].bias.data = torch.rand(hiddenDim) + torch.randint(-INT_LIMIT, INT_LIMIT, (hiddenDim,))
        hl.append(nn.ReLU())

        for i in range(2, hiddenNum+1):
            hl.append(nn.Linear(hiddenDim, hiddenDim))
            hl[-1].weight.data = torch.rand(hiddenDim, hiddenDim) + torch.randint(-INT_LIMIT, INT_LIMIT, (hiddenDim, hiddenDim))
            hl[-1].bias.data = torch.rand(hiddenDim) + torch.randint(-INT_LIMIT, INT_LIMIT, (hiddenDim,))
            hl.append(nn.ReLU())

        self.hiddenLayers = nn.Sequential(*hl)

        self.outputLayer = nn.Linear(hiddenDim, 1)
        self.outputLayer.weight.data = torch.rand(1, hiddenDim) + torch.randint(-INT_LIMIT, INT_LIMIT, (1, hiddenDim))
        self.outputLayer.bias.data = torch.rand(1) + torch.randint(-INT_LIMIT, INT_LIMIT, (1,))

    def forward(self, x):
        y = self.hiddenLayers(x)
        y = nn.functional.hardtanh(self.outputLayer(y), min_val=0, max_val=1)
        return y

