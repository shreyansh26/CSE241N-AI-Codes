import numpy as np
from random import randrange

import model

def eval_numerical_gradient(f, x, verbose=False, h=0.00001):
  """ 
  a naive implementation of numerical gradient of f at x 
  - f should be a function that takes a single argument
  - x is the point (numpy array) to evaluate the gradient at
  """ 

  fx = f(x) # evaluate function value at original point
  grad = np.zeros_like(x)
  # iterate over all indexes in x
  it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  while not it.finished:

    # evaluate function at x+h
    ix = it.multi_index
    oldval = x[ix]
    x[ix] = oldval + h # increment by h
    fxph = f(x) # evalute f(x + h)
    x[ix] = oldval - h
    fxmh = f(x) # evaluate f(x - h)
    x[ix] = oldval # restore

    # compute the partial derivative with centered formula
    grad[ix] = (fxph - fxmh) / (2 * h) # the slope
    if verbose:
      print(ix, grad[ix])
    it.iternext() # step to next dimension

  return grad

# define an arbitrary neural network
layer_sizes = [2, 4, 2, 4, 5]
no_layer = len(layer_sizes) - 1
batch_size = 16

model_nn = model.neural_network(layer_sizes)

# create a single sample example
X = np.random.randn(layer_sizes[0], batch_size)
Y = np.random.randint(0, layer_sizes[-1], batch_size)

# analytical evaluation
Y_hat, loss = model_nn.forward(X, Y)
model_nn.compute_gradients()

# numerical evaluation
grad_W = [None] * (no_layer+1)
grad_b = [None] * (no_layer+1)

f = lambda W: model_nn.forward(X, Y)[1]

for i in range(no_layer, 0, -1):
  grad_W[i] = eval_numerical_gradient(f, model_nn.W[i])
  grad_b[i] = eval_numerical_gradient(f, model_nn.b[i])

# check if implementation is correct
FLAG = True
for i in range(1, no_layer+1):
  print('gradients for W{} match?: {}'.format(i, np.allclose(grad_W[i], model_nn.grad_W[i])))
  print('gradients for b{} match?: {}'.format(i, np.allclose(grad_b[i], model_nn.grad_b[i])))
  
  if (np.allclose(grad_W[i], model_nn.grad_W[i]) == False or
      np.allclose(grad_b[i], model_nn.grad_b[i]) == False):
    FLAG = False
    break
  
if FLAG == True:
  print("Implementation is working correctly!")
else:
  print("WRONG implementation!")
