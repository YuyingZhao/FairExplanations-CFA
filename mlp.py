import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, arch=[], dropout=0.1):
        super(MLP, self).__init__()
        self.layers = []
        self.activation = nn.LeakyReLU()

        for i in range(1, len(arch)):
            self.layers.append(nn.Linear(arch[i-1], arch[i], bias=False))
            if i != len(arch)-1:
                self.layers.append(self.activation)
                self.layers.append(nn.Dropout(dropout))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        hidden_emb = self.layers[0](x)
        h = hidden_emb
        for layer in self.layers[1:]:
            h = layer(h)
        return hidden_emb, h

    def predict_proba(self, x):
        x = torch.tensor(x).float().cuda()
        h = x
        for layer in self.layers:
            h = layer(h)
        h = nn.Softmax(dim=1)(h) # transform to probability
        return h
