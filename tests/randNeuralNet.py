INT_LIMIT = 1

import torch
from torch import nn

class RandPwlNeuralNet(nn.Module):

    def __init__(self, inputDim, hiddenDim, hiddenNum, outputDim = 1):
        super(RandPwlNeuralNet, self).__init__()

        hl = []

        hl.append(nn.Linear(inputDim, hiddenDim))
        hl[-1].weight.data = torch.rand(hiddenDim, inputDim) + torch.randint(-INT_LIMIT, INT_LIMIT+1, (hiddenDim, inputDim))
        hl[-1].bias.data = torch.rand(hiddenDim) + torch.randint(-INT_LIMIT, INT_LIMIT+1, (hiddenDim,))
        hl.append(nn.ReLU())

        for i in range(2, hiddenNum+1):
            hl.append(nn.Linear(hiddenDim, hiddenDim))
            hl[-1].weight.data = torch.rand(hiddenDim, hiddenDim) + torch.randint(-INT_LIMIT, INT_LIMIT+1, (hiddenDim, hiddenDim))
            hl[-1].bias.data = torch.rand(hiddenDim) + torch.randint(-INT_LIMIT, INT_LIMIT+1, (hiddenDim,))
            hl.append(nn.ReLU())

        self.hiddenLayers = nn.Sequential(*hl)

        self.outputLayer = nn.Linear(hiddenDim, outputDim)
        self.outputLayer.weight.data = torch.rand(outputDim, hiddenDim) + torch.randint(-INT_LIMIT, INT_LIMIT+1, (outputDim, hiddenDim))
        self.outputLayer.bias.data = torch.rand(outputDim) + torch.randint(-INT_LIMIT, INT_LIMIT+1, (outputDim,))

    def forward(self, x):
        y = self.hiddenLayers(x)
        y = nn.functional.hardtanh(self.outputLayer(y), min_val=0, max_val=1)
        return y

