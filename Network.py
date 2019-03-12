import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self,inputDims, layerDims, outputDims, batch_normed=False, net_type="Actor"):

        super(MLP, self).__init__()
        self.net_type = net_type

        if batch_normed:
            self.initLayer = nn.BatchNorm1d(inputDims)
            self.initLayer.weight.data.fill_(1)
            self.initLayer.bias.data.fill_(0)

        else:
            self.initLayer = lambda x : x

        self.processingLayers = []
        self.layerDims = layerDims
        self.layerDims.insert(0,inputDims)
        self.layerDims.append(outputDims)

        for idx in range(len(self.layerDims)-1):
            self.processingLayers.append(nn.Linear(self.layerDims[idx], self.layerDims[idx+1]))

        list_param = []
        for a in self.processingLayers:
            list_param.extend(list(a.parameters()))

        self.LayerParams = nn.ParameterList(list_param)

    def forward(self, inputs):

        out = inputs
        for layers in self.processingLayers[:-1]:
            out = layers(out)
            out = F.relu(out)

        out = self.processingLayers[-1](out)
        if self.net_type == "Actor":
            out = F.tanh(out)

        return out


