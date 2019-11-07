from neural_network import *
import torch


class Layer(object):

    def __init__(self, in_dims, out_dims, activation, input, has_bias=True):
        self.weights = torch.randn(out_dims, in_dims).cuda()
        self.weight_grad = torch.randn(self.weights.shape).cuda()
        self.w_shape = self.weights.shape
        self.bias = torch.zeros(out_dims, 1).cuda() if has_bias else None
        self.bias_grad = torch.randn(self.bias.shape).cuda() if has_bias else None
        self.b_shape = self.bias.shape if has_bias else None
        self.has_bias = has_bias
        self.activation = activation
        self.input = input

    def __call__(self, input):
        return self.forward(input)

    def forward(self, input):
        has_bias = self.has_bias
        input = (input.reshape(input.shape[0], 1) if len(input.shape) == 1
                                                  else input)
        linear = torch.mm(self.weights, input)
        linear = linear + self.bias if has_bias else linear
        return self.activation(linear)

    def zero_grad(self):
        has_bias = self.has_bias
        self.weight_grad = torch.zeros(self.w_shape).cuda()
        self.weight_grad = torch.zeros(self.b_shape).cuda() if has_bias else None
        self.activation.zero_grad()

    def backward(self, lr, optimizer=None):
        raise NotImplementedError

    def initialize(self):
        raise NotImplementedError


class FCLayer(Layer):

    def __init__(self, in_dims, out_dims, activation, has_bias=True):
        super().__init__(in_dims, out_dims, activation, input, has_bias=True)

    def backward(self, lr, optimizer=None):
        if optimizer:
            optimizer.grad_update(self.weights, lr, self.input, self.w_shape)
        else:
            self.update_grad(self.input)
            self.weights -= lr * self.weight_grad
            if self.has_bias:
                self.update_grad()
                self.bias -= lr * self.bias_grad

    def update_grad(self, input):
        D = self.w_shape[0]
        stack = (input, ) * D
        input = torch.stack(stack, dim=0)
        grad = torch.zeros(self.w_shape, self.w_shape[-1])
        self.weight_grad = torch.bmm(grad, input).reshape(self.w_shape)
        if self.has_bias:
           self.bias_grad = torch.zeros(self.bias.shape)

    def initialize(self):
        self.weight_grad = torch.randn(self.weights.shape)
        if self.has_bias: self.bias = torch.zeros(self.bias.shape)


class ConvLayer(Layer):

    def __init__(self, in_dims, out_dims, activation,
                 kernel_size=3, channels=1, has_bias=True):
        super().__init__(in_dims, out_dims,
                                      activation, has_bias=True)
        self.kernel = torch.zeros(channels, kernel_size, kernel_size)

    def __call__(self, input):
        self.forward(input)

    def forward(self, input):
        pass

    def backward(self, lr, optimizer=None):
        pass

    def initialize(self):
        pass

if __name__ == '__main__':
    sensitivity = 1e-8
    print('Beginning testing...')
    print(torch.__version__)
    test = FCLayer(5, 4, lambda x: x)
    x = torch.randn(5).reshape(5, 1).cuda()
    print(test(x).shape)
    print(torch.mm(test.weights, x).shape)
    a = test(x)
    b = torch.mm(test.weights, x)
    assert torch.all(torch.lt(torch.abs(a - b), sensitivity))
    assert test.weights.shape == test.weight_grad.shape
    test.backward(0.03)
    assert test.weights.shape == test.weight_grad.shape
    print('Testing finished.')
    # ADD XOR test.

