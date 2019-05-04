import torch

from neural_network import *


class Activation():

    def __init__(self, activation, derivative):
        self.activation = activation
        self.derivative = derivative

    def __call__(self, input):
        return self.forward(input)

    def forward(self, input):
        return self.activation(input)

    def backward(self, input):
        return self.derivative(input)


class ReLU(Activation):

    def __init__(self):
        super(ReLU, self).__init__(activation=lambda x: x.clamp(min=0),
                                         derivative=lambda x: (x > 0).float())

    def __call__(self, input):
        return self.forward(input)

    def forward(self, input):
        return self.activation(input)

    def backward(self, input):
        return self.derivative(input)



if __name__ == '__main__':
    print("Beginning Testing...")
    test1 = ReLU()
    test2 = Activation(lambda x: x.clamp(min=0),
                       lambda x: (x > 0).float())
    vector = torch.randn(3, 2)
    print(vector)
    print(test1(vector))
    print(test2(vector))
    print(torch.all(test1(vector)==test2(vector)))
    print(test1.backward(vector))
    print(test2.backward(vector))
    print(torch.all(test1.backward(vector)==test2.backward(vector)))
