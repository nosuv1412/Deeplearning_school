import numpy as np

#f(x) = x^2 + 5sin(x)
def f(x):
    return x**2 + 5*np.sin(x)

# f'(x) = 2x + 5cos(x)
def grad(x):
    return 2*x + 5*np.cos(x)

def GD(x0, learning_rate):
    x = [x0]
    for u in range(100):
        x_new = x[-1] - learning_rate * grad(x[-1])
        if abs(grad(x_new)) < 1e-3:
            break
        x.append(x_new)
    return x

x = GD(3, 0.1)
print("x = {:.2f}, f(x) = {:.2f}, f'(x) = {:.2f}".format(x[-1], f(x[-1]), grad(x[-1])))