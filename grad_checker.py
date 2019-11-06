import torch

from neural_network import *


def grad_checker(function, tensor, h=1e-4, sensitivity=1e-3):
    """
    Partial code taken from Stanford's CS224N starter code.

    The function calculates the symmetric difference quotient and checks that
    the gradient is within the sensitivity specified.

    Args:
        function: The function whose gradient will be checked.
        tensor: The input to the function.
        h: In the formula:
            $$
            \frac{\partial f}{\partial x} = \frac{f(x + h*{\bf 1}) - f(x - h *
            {\bf 1})}{2 * h}
            $$
            where $f$ denotes function and ${\bf 1}$ is the column vector
            whose entries are all $1$.
            The argument h is $h$ in the above formula.
        sensitivity: Checks that the difference between the function gradient
            evaluated at the tensor and the symmetric difference quotient are
            within the desired sensitivity.
    """

    tensor = tensor.double()

    fx, grad = function.activation, function.derivative
    numels = grad(tensor).numel()

    tensor_shape = tensor.shape
    tensor = tensor.reshape(tensor.numel())

    grad_list = []
    diff_list = []

    for element in tensor:
        low_f = fx(element - h)
        high_f = fx(element + h)
        numgrad = (high_f - low_f) / (2 * h)
        grad_list.append(numgrad)

        reldiff = torch.abs(numgrad - grad(element)) / max(1,
                                                           torch.abs(numgrad),
                                                           torch.abs(grad(element)))
        diff_list.append(reldiff)

        if reldiff > sensitivity:
            print("Gradient check failed")
            print("Check first failed at {}".format(element))
            print(reldiff)
            return


    print("Gradient check passed.")


if __name__ == '__main__':
    print("Beginning Testing")

    test = activation.Activation(activation=lambda x: x.sum() ** 2,
                                 derivative=lambda x: 2 * x)

    grad_checker(test, torch.tensor([123.456]))
    grad_checker(test, torch.randn(3))

    grad_checker(test, torch.randn(2, 3, 4))
    test_tensor = torch.randn(2, 3, 2)
