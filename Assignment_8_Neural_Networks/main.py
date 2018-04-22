import model
import utils

import os
import time
import numpy as np
import matplotlib.pyplot as plt


####################################################
# MODIFY HYPERPARAMETERS (START)
####################################################
# hyperparameters
# You should tune these hyper parameters to maximize precision
# Observe how increasing the size and number of layers
# results in overfitting on the dataset

# UNCOMMENT THE NEXT THREE LINES AND FILL IN YOUR HYPERPARAMETERS
no_epochs = 1000
lr = 0.00001
layer_sizes = [2, 25, 25, 25, 3] # list specifying network architecture

####################################################
# MODIFY HYPERPARAMETERS (END)
####################################################

no_layers = len(layer_sizes)+1

# create a neural network model
model_nn = model.neural_network(layer_sizes)

begin_time = time.time()

# The data is a 2D spiral. So the input is a 2 dimensional point, and they are classified into three
# classes
# data taken from http://cs231n.github.io/neural-networks-case-study/

# load the data
X_train = np.load('train_X.npy')
Y_train = np.load('train_Y.npy')

X_val = np.load('val_X.npy')
Y_val = np.load('val_Y.npy')

# make the dimensions of data as [no_features x batch_size]
X = X_train.T
Y = Y_train

# UNCOMMENT TO VISUALIZE THE DATA
# fig = plt.figure()
# plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
# plt.xlim([-1,1])
# plt.ylim([-1,1])
# plt.show()

for epoch in range(no_epochs):
    # training
    losses = utils.AverageMeter() 
    precision = utils.AverageMeter() 

    # compute output and loss
    output, loss = model_nn.forward(X, Y)

    # compute the gradients
    model_nn.compute_gradients()

    # perform gradient updates
    model_nn.gradient_descent(lr)

    # measure accuracy and record loss   
    prec = utils.accuracy(output, Y)
    losses.update(loss, Y.shape[0])
    precision.update(prec, Y.shape[0])

    print('-----E[{epoch:03d}]\t'
          'Time {timer:.1f}\t'
          'Loss {loss.val:.1f}({loss.avg:.1f})\t'
          'Prec {precision.val:.3f}'
          '({precision.avg:.3f})'.format(epoch=epoch,
                                         timer=time.time()-begin_time,
                                         loss=losses, precision=precision))


    # validation
    # compute output and loss
    output, loss = model_nn.forward(X_val.T, Y_val)
    prec = utils.accuracy(output, Y_val)
    print('Validation Precision: {}'.format(prec))

# plot the results
# SQUARE represent the validation examples
# CROSS represent the training examples

X = X_train
y = Y_train
# plot the resulting classifier
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = model_nn.forward(np.c_[xx.ravel(), yy.ravel()].T,
                     np.ones(np.c_[xx.ravel(), yy.ravel()].shape[0],
                             dtype='int8'))[0]
Z = np.argmax(Z, axis=0)
Z = Z.reshape(xx.shape)
fig = plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, s=20, cmap=plt.cm.Spectral, alpha=0.6, marker='x')
plt.scatter(X_val[:, 0], X_val[:, 1], c=Y_val, s=20, cmap=plt.cm.Spectral, marker='s', alpha=19)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()


# store the hyperparameters
model_nn.store_parameters('model_nn.pkl')
