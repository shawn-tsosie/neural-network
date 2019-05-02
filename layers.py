import torch
import data
import activations


class Layer(object):

    def __init__(self, in_dims, out_dims, activation, has_bias=True):
        self.weights = torch.randn(out_dims, in_dims).cuda()
        self.weight_grad = torch.randn(self.weights.shape).cuda()
        self.w_shape = self.weights.shape
        self.bias = torch.zeros(out_dims, 1).cuda() if has_bias else None
        self.bias_grad = torch.randn(self.bias.shape).cuda() if has_bias else None
        self.b_shape = self.bias.shape if has_bias else None
        self.has_bias = has_bias
        self.activation = activation

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
        super(FCLayer, self).__init__(in_dims, out_dims,
                                      activation, has_bias=True)

    def backward(self, lr, optimizer=None):
        if optimizer:
            pass
        else:
            pass

    def initialize(self):
        pass


class ConvLayer(Layer):

    def __init__(self):
        pass

if __name__ == '__main__':
    print('Beginning testing...')
    print(torch.__version__)
    test = FCLayer(5, 4, lambda x: x)
    x = torch.randn(5)
    print(x)
    x = (x.reshape(x.shape[0], 1) if len(x.shape) == 1 else x)
    print(x)
    print(torch.mm(test.weights, x).shape)
    print(test.bias)
    print(test.weights)
    print(test(x).shape)

