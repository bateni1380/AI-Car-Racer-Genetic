import torch
import torch.nn as nn

class CarNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CarNet, self).__init__()
        self.layers = []
        self.layers.append(nn.Linear(input_size, hidden_size))
        #self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.layers.append(nn.Linear(hidden_size, output_size))
        self.depth = len(self.layers)

    def forward(self, x):
        for i in range(self.depth-1):
            x = torch.relu(self.layers[i](x))
        x = self.layers[self.depth-1](x)
        return torch.sigmoid(x)

    def to_list(self):
        l = torch.tensor([])
        for i in range(self.depth):
            l = torch.cat((l, self.layers[i].weight.data.view(-1)))
        return l.tolist()

    def update_from_list(self, l):
        t1, t2 = 0, 0
        for i in range(self.depth):
            t1, t2=t2, t2 + self.layers[i].weight.numel()
            self.layers[i].weight.data=torch.tensor(l[t1:t2]).view(self.layers[i].weight.data.shape)
        return self