"""Two Layer Network."""
# pylint: disable=invalid-name
import numpy as np


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension
    of N, a hidden layer dimension of H, and performs classification over C
    classes. We train the network with a softmax loss function and L2
    regularization on the weight matrices. The network uses a ReLU nonlinearity
    after the first fully connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each
    class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each
          y[i] is an integer in the range 0 <= y[i] < C. This parameter is
          optional; if it is not passed then we only return scores, and if it is
          passed then we instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c]
        is the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of
          training samples.
        - grads: Dictionary mapping parameter names to gradients of those
          parameters  with respect to the loss function; has the same keys as
          self.params.
        """
        # pylint: disable=too-many-locals
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, _ = X.shape

        # Compute the forward pass
        scores = None
        ########################################################################
        # TODO: Perform the forward pass, computing the class scores for the   #
        # input. Store the result in the scores variable, which should be an   #
        # array of shape (N, C).                                               #         
        ########################################################################
        S1=np.maximum(np.dot(X,W1)+b1,0)
        scores=np.dot(S1,W2)+b2
        ########################################################################
        #                              END OF YOUR CODE                        #
        ########################################################################

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = None
        ########################################################################
        # TODO: Finish the forward pass, and compute the loss. This should     #
        # include both the data loss and L2 regularization for W1 and W2. Store#
        # the result in the variable loss, which should be a scalar. Use the   #
        # Softmax classifier loss. So that your results match ours, multiply   #
        # the regularization loss by 0.5                                       #
        ########################################################################
        scores-=np.max(scores,axis=1,keepdims=True)
        P=np.exp(scores)/np.sum(np.exp(scores),axis=1,keepdims=True)
        P_k=P[range(N),y]
        data_loss=-np.sum(np.log(P_k))/N
        loss=data_loss+0.5*reg*(np.sum(W1*W1)+np.sum(W2*W2))
        ########################################################################
        #                              END OF YOUR CODE                        #
        ########################################################################

        # Backward pass: compute gradients
        grads = {}
        ########################################################################
        # TODO: Compute the backward pass, computing the derivatives of the    #
        # weights and biases. Store the results in the grads dictionary. For   #
        # example, grads['W1'] should store the gradient on W1, and be a matrix#
        # of same size                                                         #
        ########################################################################
        P[range(N),y]-=1
        dscores=P/N
        grads['W2']=np.dot(S1.T,dscores)+reg*W2
        grads['b2']=np.sum(dscores,axis=0)
        ## compute the dLoss/dOutput_layer1
        
        dHidden=np.dot(dscores,W2.T)
        hidden_mask=np.zeros_like(dHidden)
        hidden_mask[S1>0]=1
        dLayer1=hidden_mask*dHidden
        ##compute dLoss/db2=dLayer*1
        ##dLoss/dW1=dLayer
        grads['b1']=np.sum(dLayer1,axis=0)
        grads['W1']=np.dot(X.T,dLayer1)+reg*W1

        ########################################################################
        #                              END OF YOUR CODE                        #
        ########################################################################

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=1e-5, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means
          that X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning
          rate after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        # pylint: disable=too-many-arguments, too-many-locals
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train // batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            ####################################################################
            # TODO: Create a random minibatch of training data and labels,     #
            # storing hem in X_batch and y_batch respectively.                 #
            ####################################################################
            ind=np.random.choice(num_train,batch_size)
            X_batch=X[ind]
            y_batch=y[ind]

            ####################################################################
            #                             END OF YOUR CODE                     #
            ####################################################################

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            ####################################################################
            # TODO: Use the gradients in the grads dictionary to update the    #
            # parameters of the network (stored in the dictionary self.params) #
            # using stochastic gradient descent. You'll need to use the        #
            # gradients stored in the grads dictionary defined above.          #
            ####################################################################
            self.params['W1']-=learning_rate*grads['W1']
            self.params['b1']-=learning_rate*grads['b1']
            self.params['W2']-=learning_rate*grads['W2']
            self.params['b2']-=learning_rate*grads['b2']

            ####################################################################
            #                             END OF YOUR CODE                     #
            ####################################################################

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each
          of the elements of X. For all i, y_pred[i] = c means that X[i] is
          predicted to have class c, where 0 <= c < C.
        """
        y_pred = None

        ########################################################################
        # TODO: Implement this function; it should be VERY simple!             #
        ########################################################################
        W1,b1=self.params['W1'],self.params['b1']
        W2,b2=self.params['W2'],self.params['b2']
        hidden_layer=np.maximum(0,np.dot(X,W1)+b1)
        scores=np.dot(hidden_layer,W2)+b2
        y_pred=np.argmax(scores,axis=1)

        ########################################################################
        #                              END OF YOUR CODE                        #
        ########################################################################

        return y_pred


def neuralnetwork_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    best_net = None # store the best model into this 
    ############################################################################
    # TODO: Tune hyperparameters using the validation set. Store your best     #
    # trained model in best_net.                                               #
    #                                                                          #
    # To help debug your network, it may help to use visualizations similar to #
    # the  ones we used above; these visualizations will have significant      #
    # qualitative differences from the ones we saw above for the poorly tuned  #
    # network.                                                                 #
    #                                                                          #
    # Tweaking hyperparameters by hand can be fun, but you might find it useful#
    # to  write code to sweep through possible combinations of hyperparameters #
    # automatically like we did on the previous exercises.                     #
    ############################################################################
    input_size=32*32*3
    hidden_size=[100,150,300]
    num_classes=10
    learning_rate=[1e-3,1.5e-3]
    batch_size=[200,300]
    best_val=-1
    for hs in hidden_size:
        for lr in learning_rate:
            for bs in batch_size:
                net=TwoLayerNet(input_size, hs, num_classes)
                stats=net.train(X_train, y_train, X_val, y_val,
                                num_iters=2000, batch_size=bs,learning_rate=lr, 
                                learning_rate_decay=0.95,reg=0.5, verbose=False)
                
                acc_val = (net.predict(X_val) == y_val).mean()
                print('hidden_size:%f, learning rate: %f, batch size:%f, valid accuracy:%f'%(hs,lr,bs,acc_val))
                if acc_val>best_val:
                    best_val=acc_val
                    best_net=net
# =============================================================================
#     hidden_size=300
#     input_size=32*32*3
#     num_classes=10
#     learning_rate=1.5e-3
#     batch_size=300
#     net=TwoLayerNet(input_size, hidden_size, num_classes)
#     stats=net.train(X_train, y_train, X_val, y_val,
#     num_iters=2000, batch_size=300,learning_rate=1.5e-3, 
#     learning_rate_decay=0.95,reg=0.5, verbose=False)
#                 
#     acc_val = (net.predict(X_val) == y_val).mean()
#     best_val=acc_val
#     best_net=net
#     print('hidden_size:%f, learning rate: %f, batch size:%f, valid accuracy:%f'%(hidden_size,learning_rate,batch_size,acc_val))
#                 
# =============================================================================
    

    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################
    return best_net
