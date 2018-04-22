import numpy as np
import pickle

class neural_network:
    '''
    neural network class is used to create the neural classification model that is used in the 
    assignment. 
    It has multiple data elements:
    l: number of layers of the neural network (hidden layers + output layer)
    W: list that stores the weight matrix (a numpy array) for each layer of the network 
       (list is indexed from 1 to l)
    b: list that stores the bias vector for each layer of the network (list is index from 1 to l)
    h: list of the outputs of the hidden layers
    a: list of the pre--activation function outputs of the layers
    grad_b: list of gradient of loss function with respect to the bias
    grad_W: list of gradient of loss function with respect to the weight matrices

    It has the following functions:
    forward(): it calculates the class scores by the model for given input points
    compute_gradient(): it computes all the W_grad and b_grad with the help of
                        backpropogation algorithm 
    gradient_descent(): it updates all the parameters of the network by using gradient descent
    loss(): it computes the cross entropy loss between the class scores and the actual 
            values of class
    grad_loss(): computes the gradient of the loss with respect to Y_hat
    activation_fn(): the activation function, applied elementwise to the input
    grad_activation_fn(): gradient of activation function, applied elementwise to the input
    store_parameters()
    load_parameters()
    '''
    
    def __init__(self, layer_dimension_list=None, load_from_file=False, file_name=None):
        '''This function defines the neural network architecture. 
        It also initializes weight matrices self.W[i] and bias self.b[i].
        self.h[], self.a[]: are lists initialized with 'None' and correspond to the layer outputs
        after and before activations respectively (as given in hint.pdf). These are to be 
        used in forward() function below.

        self.grad_b[] and self.grad_W[]: lists are also initialized with 'None'. They should 
        contain the gradients of corresponding parameters. They are to be assigned in
        backward() function below.

        Input:
        layer_dimension_list: a list of number of neurons in each layer.
        eg: [4, 3, 2] specifies a neural network with 
                      4 input neurons, 
                      3 neurons in the hidden layer and 
                      2 neurons in the output layer.

        NOTE: You can index all the lists from 1 to l, instead of (0 to l-1). So the indexing
              can more closely match that used in Section 4 of 'hint.pdf'.
        '''

        if load_from_file:
            net_params = self.load_weights(file_name)
            
            self.layer_dimension_list = net_params['layers']
            self.l = len(self.layer_dimension_list) - 1

            self.W = net_params['W']
            self.b = net_params['b']
            
        else:
            self.l = len(layer_dimension_list) - 1
            self.layer_dimension_list = layer_dimension_list
            
            # initialize the weight matrices
            self.W = [None]
            self.b = [None]
            for i in range(self.l):
                self.W.append(np.random.randn(layer_dimension_list[i+1], layer_dimension_list[i]))
                self.b.append(np.random.randn(layer_dimension_list[i+1], 1))
                
        l = self.l
                
        self.h = [None]*(l+1)
        self.a = [None]*(l+1)

        self.grad_b = [None]*(l+1)
        self.grad_W = [None]*(l+1)

    def forward(self, X, Y):
        '''This function should compute Y_hat (a numpy array containing output scores for 
        each class, of size [no_of_classes x batch_size]) and J (final_loss, a scalar) 
        on the input features X and target labels Y. 
        X is a numpy array of size [no_feature x batch_size] and Y is a numpy array of size
        [batch_size].
        '''
        self.Y = Y
        self.Y_hat = None
        final_loss = None

        # begin the forward pass algorithm
        
        ####################################################
        # YOUR CODE BEGINS HERE
        ####################################################
        self.h[0]=X

        i = 1
        while i<self.l:
            self.a[i] = self.W[i].dot(self.h[i-1]) + self.b[i].reshape(-1,1)
            self.h[i] = self.activation_fn(self.a[i])
            i+=1

        self.a[self.l] = self.W[self.l].dot(self.h[self.l-1]) + self.b[self.l].reshape(-1,1)
        self.Y_hat = self.a[self.l]

        final_loss = np.sum(self.loss(self.Y, self.Y_hat))
        ####################################################
        # YOUR CODE ENDS HERE
        ####################################################
        
        return self.Y_hat, final_loss

    def compute_gradients(self):
        '''This function evaluates the values of self.grad_W[i] and self.grad_b[i] for i = 1, ..., l
        using backpropogation algorithm.
        This function doesn't return anything.
        '''

        ####################################################
        # YOUR CODE BEGINS HERE
        ####################################################
        grad = self.grad_loss(self.Y, self.Y_hat)

        self.grad_W[self.l] = grad.dot(self.h[self.l-1].T)
        self.grad_b[self.l] = np.sum(grad, axis=1, keepdims=True)

        grad = self.W[self.l].T.dot(grad)

        i = self.l-1
        while i>0:
            grad = grad*self.grad_activation_fn(self.a[i])
            self.grad_W[i] = grad.dot(self.h[i-1].T)
            self.grad_b[i] = np.sum(grad, axis=1, keepdims=True)
            grad = self.W[i].T.dot(grad)
            i-=1

        ####################################################
        # YOUR CODE ENDS HERE
        ####################################################

    def gradient_descent(self, lr):
        '''Performs the gradient descent update for each weight matrix 
        and bias vector.
        '''
        for i in range(1, self.l+1):
            self.W[i] -= lr * self.grad_W[i]
            self.b[i] -= lr * self.grad_b[i]

    def loss(self, Y, Y_hat):
        '''Return the cross-entropy loss for each sample. Return type is a numpy array of 
        length = batch_size. ith component refers to the loss of ith sample.
        '''
        log_C = -np.max(Y_hat)
        
        exp_Y_hat = np.exp(Y_hat + log_C)
        denom = exp_Y_hat.sum(0)

        J = -np.log(exp_Y_hat[Y, range(exp_Y_hat.shape[1])] / denom)

        return J

    def grad_loss(self, Y, Y_hat):
        '''Return the gradient of the cross-entropy loss, with respect to Y_hat.
        Return type is a numpy array of dimensions same as Y_hat. ith component refers 
        to the loss of ith sample.
        '''
        log_C = -np.max(Y_hat)
                
        exp_Y_hat = np.exp(Y_hat + log_C)
        denom = exp_Y_hat.sum(0)

        grad_J = exp_Y_hat / denom
        grad_J[Y, range(Y_hat.shape[1])] = grad_J[Y, range(Y_hat.shape[1])] - 1

        return grad_J

    def activation_fn(self, X):
        '''This function should apply elementwise an activation function (like tanh()) to the input. 
        Input would be a 2D numpy array. Output would be a numpy array of similar dimensions.
        '''
        return np.tanh(X)

    def grad_activation_fn(self, X):
        '''This function would apply elementwise the gradient of the activation function with respect
        to its input. (like tanh())  to the input. 
        Input would be a 2D numpy array. Output would be a numpy array of similar dimensions.
        '''
        return  1-np.tanh(X)**2
    
    def store_parameters(self, file_name):
        '''Store the parameters: layer_dimension_list, self.W and self.b in 'model_nn.pkl'
        '''        
        net_params = dict()
        net_params['layers'] = self.layer_dimension_list
        net_params['W'] = self.W
        net_params['b'] = self.b

        out_file = open(file_name, 'wb')
        pickle.dump(net_params, out_file)
        out_file.close()

    
    def load_weights(self, file_name):
        '''Load the parameters: layer_dimension_list, self.W and self.b from 'model_nn.pkl'.
        These will be used to initialize the model architecture and put in pre--trained weights.
        '''            
        in_file = open(file_name, 'rb')
        net_params = pickle.load(in_file)
        in_file.close()

        return net_params

