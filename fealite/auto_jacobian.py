import torch
from torch.autograd.gradcheck import zero_gradients


def alpha(x):
    return torch.exp(x)


def f(x):
    # Matrix getting constructed
    k = torch.zeros((x.shape[0], x.shape[0]))

    # loop over some random 3 dimensional vectors
    for element in torch.randint(0, x.shape[0], (x.shape[0], 3)):

        # select 3 values from x
        x_ijk = torch.tensor([[1. if n == e else 0 for n in range(x.shape(0))] for e in element]) @ x
        norm = torch.norm(
            x_ijk @ torch.stack((torch.tanh(element.float() + 4), element.float() - 4)).t()
        )

        m = torch.rand(3, 3)

        # alpha is an arbitrary differentiable function R -> R
        alpha_value = alpha(norm)

        n = m.shape[0]
        for i in range(n):
            for j in range(n):
                k[element[i], element[j]] += m[i, j] * alpha_value

    print(k)
    return k @ x


x = torch.rand(4, requires_grad=True)
print(x, '\n')
y = f(x)
print(y, '\n')
grads = []
for val in y:
    val.backward(retain_graph=True)
    grads.append(x.grad.clone())
    zero_gradients(x)
if __name__ == '__main__':
    print(torch.stack(grads))
