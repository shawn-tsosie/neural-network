import torch

from functools import partial
from neural_network import *


class Activation():

    def __init__(self, activation, derivative):
        self.activation = activation
        self.derivative = derivative

    def __call__(self, tensor):
        return self.forward(tensor)

    def forward(self, tensor):
        return self.activation(tensor)

    def backward(self, tensor):
        return self.derivative(tensor)


class ReLU(Activation):

    def __init__(self):
        super().__init__(activation=lambda x: x.clamp(min=0),
                         derivative=lambda x: (x > 0).float())


def prelu_func(tensor, alpha):
    return x if x > 0 else alpha * x

def prelu_grad(tensor, a):
    pass


class PReLU(Activation):

    def __init__(self, parameter):
        self.parameter = parameter
        super().__init__(activation=partial(prelu_func, a=self.parameter),
                         derivative=partial(prelu_grad, a=self.parameter))


def softmax_func(tensor):
    """
    Based off of the starter code from Stanford's CS224N
    """
    original_shape = tensor.shape

    if len(tensor.shape) > 1:
        D = tensor.shape[0]
        tensor_max = torch.max(tensor, dim=1).reshape(D, 1)
        softmax = tensor - tensor_max
        softmax = torch.exp(softmax)
        softmax /= softmax.sum(dim=1).reshape(D, 1)
        tensor = softmax
    else:
        tensor_max = torch.max(tensor)
        softmax = tensor - tensor_max
        softmax = torch.exp(softmax)
        softmax /= softmax.sum()
        tensor = softmax

    assert tensor.shape == original_shape
    return tensor

def softmax_grad(tensor):
    original_shape = tensor.shape

    if len(tensor.shape) > 1:
        D = tensor.shape[0]
        tensor_max = torch.max(tensor, dim=1).reshape(D, 1)
        softmax = tensor - tensor_max
        softmax = torch.exp(softmax)
        softmax /= softmax.sum(dim=1).reshape(D, 1)
        tensor = softmax
        tensor_grad = softmax
    else:
        tensor_max = torch.max(tensor)
        softmax = tensor - tensor_max
        softmax = torch.exp(softmax)
        softmax /= softmax.sum()
        tensor = softmax.reshape(tensor.shape, 1)
        tensor_grad = tensor.diag(tensor) - tensor.matmul(tensor.t())

    return tensor_grad


class SoftMax(Activation):

    def __init__(self):
        super().__init__(activation=softmax_func,
                         derivative=softmax_grad)


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
