import torch
from torch import nn


class GST(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output


gst = GST()
x = torch.tensor(1.0)
output = gst(x)
print(output)
