"""Linear Softmax Classifier."""
# pylint: disable=invalid-name
import numpy as np

from .linear_classifier import LinearClassifier


def cross_entropoy_loss_naive(W, X, y, reg):
    """
    Cross-entropy loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # pylint: disable=too-many-locals
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient using explicit     #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################
    
    S=np.dot(X,W)
    N,C=X.shape[0],W.shape[1]
    L=np.zeros(N)
    for i in range(N):
        Ct=-np.max(S[i,:])
        P=np.exp(S[i,y[i]]+Ct)/np.sum(np.exp(S[i,:]+Ct))
        L[i]=-np.log(P)
        ## while y[i]==j, dw-=X[i]
        dW[:,y[i]]-=X[i] 
        for j in range(C):
            dW[:,j]+=np.exp(S[i,j]+Ct)*X[i,:]/np.sum(np.exp(S[i,:]+Ct))
    loss=np.sum(L)/N+0.5*reg*np.sum(W*W)
    dW=dW/N+reg*W
    

    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


def cross_entropoy_loss_vectorized(W, X, y, reg):
    """
    Cross-entropy loss function, vectorized version.

    Inputs and outputs are the same as in cross_entropoy_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient without explicit   #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################
    pass
    S=np.dot(X,W)
    N=X.shape[0]
# =============================================================================
#     Ct=-np.max(S,axis=1,keepdims=True)
# =============================================================================
    S-=np.max(S,axis=1,keepdims=True)
    P=np.exp(S)/np.sum(np.exp(S),axis=1,keepdims=True)
    P_k=P[range(N),y[:]]
    loss=np.sum(-np.log(P_k))/N+0.5*reg*np.sum(W*W)
    

    P[range(N),y[:]]-=1
    dW=np.dot(X.T,P)
    dW=dW/N+reg*W


    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


class SoftmaxClassifier(LinearClassifier):
    """The softmax classifier which uses the cross-entropy loss."""

    def loss(self, X_batch, y_batch, reg):
        return cross_entropoy_loss_vectorized(self.W, X_batch, y_batch, reg)


def softmax_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    # results is dictionary mapping tuples of the form
    # (learning_rate, regularization_strength) to tuples of the form
    # (training_accuracy, validation_accuracy). The accuracy is simply the
    # fraction of data points that are correctly classified.
    results = {}
    best_val = -1
    best_softmax = None
    all_classifiers = []
    learning_rates = [1e-7, 5e-7, 1e-6,5e-6,1e-5]
    regularization_strengths = [1e3, 1e4, 2.5e4, 5e4]

    ############################################################################
    # TODO:                                                                    #
    # Write code that chooses the best hyperparameters by tuning on the        #
    # validation set. For each combination of hyperparameters, train a         #
    # classifier on the training set, compute its accuracy on the training and #
    # validation sets, and  store these numbers in the results dictionary.     #
    # In addition, store the best validation accuracy in best_val and the      #
    # Softmax object that achieves this accuracy in best_softmax.              #                                      #
    #                                                                          #
    # Hint: You should use a small value for num_iters as you develop your     #
    # validation code so that the classifiers don't take much time to train;   # 
    # once you are confident that your validation code works, you should rerun #
    # the validation code with a larger value for num_iters.                   #
    ############################################################################
    iters=1500
    for lr in learning_rates:
        for rs in regularization_strengths:
            softmax = SoftmaxClassifier()
            softmax.train(X_train,y_train,learning_rate=lr,reg=rs, num_iters=iters,batch_size=250,verbose=False)
            y_train_pred=softmax.predict(X_train)
            accu_train=np.mean(y_train == y_train_pred)
        
            y_val_pred=softmax.predict(X_val)
            accu_val=np.mean(y_val == y_val_pred)
        
            results[(lr,rs)]=(accu_train, accu_val)
            all_classifiers.append((softmax,accu_val,lr,rs))
            if accu_val > best_val:
                best_val=accu_val
                best_softmax=softmax
    
        

    ############################################################################
    #                              END OF YOUR CODE                            #
    ############################################################################
        
    # Print out results.
    for (lr, reg) in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
              lr, reg, train_accuracy, val_accuracy))
        
    print('best validation accuracy achieved during validation: %f' % best_val)

    return best_softmax, results, all_classifiers
