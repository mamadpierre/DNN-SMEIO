import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class AgentNet(nn.Module):

    def __init__(self, layers=[(2, 48), (48, 12), (12, 1)], num_shared=1, Activation=F.softplus, DNNEdges=[(0, 1)],
                 device="cuda"):
        super(AgentNet, self).__init__()
        self.layers = layers
        self.num_shared = num_shared
        self.Activation = Activation
        self.DNNEdges = DNNEdges
        self.fcShared = nn.ModuleDict()
        self.fcAction = nn.ModuleDict()
        for RLedge in self.DNNEdges:
            RLedge = str(RLedge)
            fcSharedLayers = []
            fcActionLayers = []
            for layer in self.layers[:self.num_shared]:
                fcSharedLayers.append(nn.Linear(layer[0], layer[1]))
                fcSharedLayers.append(nn.BatchNorm1d(layer[1]))
            self.fcShared[RLedge] = nn.Sequential(*fcSharedLayers).to(device)
            for layer in self.layers[self.num_shared:]:
                fcActionLayers.append(nn.Linear(layer[0], layer[1]))
                fcActionLayers.append(nn.BatchNorm1d(layer[1]))
            self.fcAction[RLedge] = nn.Sequential(*fcActionLayers).to(device)

            # self.fcShared[RLedge] = nn.Sequential(*[nn.Linear(layer[0], layer[1]) for layer in self.layers[:num_shared]]).to(device)
            # self.fcAction[RLedge] = nn.Sequential(*[nn.Linear(layer[0], layer[1]) for layer in self.layers[num_shared:]]).to(device)
    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        xActions = {}
        for RLedge2 in self.DNNEdges:
            RLedge = str(RLedge2)
            x2 = x
            for layer in self.fcShared[RLedge]:
                x2 = self.Activation(layer(x2))
            xAction = x2
            for layer in self.fcAction[RLedge]:
                xAction = self.Activation(layer(xAction))
            xActions[RLedge2] = xAction
        return xActions


class Teacher(object):
    def __init__(self, layers, num_shared, Activation, device, lr, DNNEdges):
        self.DNNEdges = DNNEdges
        self.device = device
        self.AgentNet = AgentNet(layers, num_shared, Activation, DNNEdges, device).to(
            self.device)
        self.optim = optim.Adam(self.AgentNet.parameters(), lr=lr)
