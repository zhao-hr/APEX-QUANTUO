import torch
import torch.nn as nn
import torch.nn.functional as F

class FM(nn.Module):
    def __init__(self, n=10, k=5):
        super().__init__()
        self.dtype = torch.float
        self.n = n
        self.k = k
        self.linear = nn.Linear(self.n, 1)
        self.V = nn.Parameter(torch.randn(self.n, self.k))
        nn.init.uniform_(self.V, -0.1, 0.1)
    
    def fm_layer(self, x):
        linear_part = self.linear(x)
        interaction_part_1 = torch.mm(x, self.V)
        interaction_part_1 = torch.pow(interaction_part_1, 2)
        interaction_part_2 = torch.mm(torch.pow(x, 2), torch.pow(self.V, 2))
        output = linear_part + 0.5 * torch.sum(interaction_part_2 - interaction_part_1, 1, keepdim=True)
        output = torch.squeeze(output, dim=1)
        return torch.sigmoid(output)

    def forward(self, x):
        return self.fm_layer(x)

if __name__ == '__main__':
    fm = FM(10, 5)
    x = torch.randn(2, 10)
    print(x)
    output = fm(x)
    print(output)
