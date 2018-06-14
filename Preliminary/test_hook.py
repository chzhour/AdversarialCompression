from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(10, 100)

    def forward(self, x):
        return self.fc(x)


def reverse_grad_hook(self, grad_input, grad_output):
    print(len(grad_input))
    print(grad_input[0].size())
    print(grad_input[1].size())
    print(grad_input[2].size())

    print(len(grad_output))
    print(grad_output[0].size())

    print(grad_input[2][0,0])    
    



def main():
    input = torch.rand(3, 10, requires_grad = True)
    model = Net()
    model.fc.register_backward_hook(reverse_grad_hook)

    output = model(input)
    output.backward(torch.rand(3, 100))

    print(model.fc.weight.grad[0,0])



if __name__ == '__main__':
    main()