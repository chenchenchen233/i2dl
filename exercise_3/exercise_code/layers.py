import numpy as np


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and width
    W. We convolve each input with F different filters, where each filter spans
    all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param) for the backward pass
    """
    out = None
    #############################################################################
    # TODO: Implement the convolutional forward pass.                           #
    # Hint: you can use the function np.pad for padding.                        #
    #############################################################################
    N,C,H,W=x.shape
    F,C,HH,WW=w.shape
    stride,pad=conv_param['stride'],conv_param['pad']
    xpad=np.pad(x,((0,0),(0,0),(pad,pad),(pad,pad)),'constant')
    ##compute the output shape
    height=(H-HH+pad*2)//stride+1
    weight=(W-WW+pad*2)//stride+1
    out=np.zeros((N,F,height,weight))
    for i in range(height):
        for j in range(weight):
            mask=xpad[:,:,i*stride:i*stride+HH,j*stride:j*stride+WW]
            for k in range(F):
                
                out[:,k,i,j]=np.sum(mask*w[k,:,:,:],axis=(1,2,3))
    out=out+(b)[None,:,None,None]##None equals to np.newaxis
                
                
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    #############################################################################
    # TODO: Implement the convolutional backward pass.                          #
    #############################################################################
    x,w,b,conv_param=cache
    N,C,H,W=x.shape
    F,_,HH,WW=w.shape
    stride,pad=conv_param['stride'],conv_param['pad']
    
    h_out=(H-HH+pad*2)//stride+1
    w_out=(W-WW+pad*2)//stride+1
    ##initialize
    dx=np.zeros_like(x)
    xpad=np.pad(x,((0,0),(0,0),(pad,pad),(pad,pad)),'constant')
    dxpad=np.zeros_like(xpad)
    dw=np.zeros_like(w)
    
    db=np.sum(dout, axis=(0,2,3))
    for i in range(N):
        for j in range(F):
            for k in range(h_out):
                for l in range(w_out):
                    x_mask=xpad[i,:,k*stride:k*stride+HH,l*stride:l*stride+WW]
                    dxpad[i,:,k*stride:k*stride+HH,l*stride:l*stride+WW] += dout[i,j,k,l]*w[j,:,:,:]
                    dw[j,:,:,:] += x_mask*dout[i,j,k,l]
    dx=dxpad[:,:,pad:-pad,pad:-pad]
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, maxIdx, pool_param) for the backward pass with maxIdx, of shape (N, C, H, W, 2)
    """
    out = None
    #############################################################################
    # TODO: Implement the max pooling forward pass                              #
    #############################################################################
    N,C,H,W=x.shape
    ph=pool_param['pool_height']
    pw=pool_param['pool_width']
    S=pool_param['stride']
    H_out=(H-ph)//S+1
    W_out=(W-pw)//S+1
    out =np.zeros((N,C,H_out,W_out))
    maxIdx=np.zeros_like(x)
    for i in range(H_out):
        for j in range(W_out):
            xmask=x[:,:,i*S:i*S+ph,j*S:j*S+pw]
            max_mask=np.max(xmask,axis=(2,3))
            out[:,:,i,j]=(max_mask)[None,None,:,:]
            Idx=(xmask==(max_mask)[:,:,None,None])
            maxIdx[:,:,i*S:i*S+ph,j*S:j*S+pw]=Idx
# =============================================================================
#     for k in range(N):
#         for l in range(C):
#             for i in range(H_out):
#                 for j in range(W_out):
#                     xmask=x[k,l,i*S:i*S+ph,j*S:j*S+pw]
#                     out[k,l,i,j]=np.max(xmask)
#                     max_mask=out[k,l,i,j]
#                     maxIdx[k,l,i*S:i*S+ph,j*S:j*S+pw]=(xmask==(max_mask)[None,None])
#             
# =============================================================================
    
            #max_mask=np.max(xmask,axis=(2,3))
            #out[:,:,i,j]=(max_mask)[None,None,:,:]
            #test=(max_mask)[:,:,None,None]
# =============================================================================
#             Idx=(xmask==(max_mask)[:,:,None,None])
#             print(Idx)
#             maxIdx[:,:,i*S:i*S+ph,j*S:j*S+pw]=Idx
# =============================================================================

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = (x, maxIdx, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    #############################################################################
    # TODO: Implement the max pooling backward pass                             #
    #############################################################################
    x,maxIdx,pool_param=cache
    N,C,H,W=x.shape
    ph=pool_param['pool_height']
    pw=pool_param['pool_width']
    S=pool_param['stride']
    H_out=(H-ph)//S+1
    W_out=(W-pw)//S+1
    dx=np.zeros_like(x)
    for i in range(H_out):
        for j in range(W_out):
            dx[:,:,i*S:i*S+ph,j*S:j*S+pw]=maxIdx[:,:,i*S:i*S+ph,j*S:j*S+pw]*(dout[:,:,i,j])[:,:,None,None]
# =============================================================================
#             a=maxIdx[:,:,i*S:i*S+ph,j*S:j*S+pw]
#             print('a=',a)
#             b=dout[:,:,i,j]
#             dx[:,:,i*S:i*S+ph,j*S:j*S+pw]=a*b
# =============================================================================
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
  """
  Forward pass for batch normalization.

  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7 implementation
  of batch normalization also uses running averages.

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

  out, cache = None, None
  if mode == 'train':
    #############################################################################
    # TODO: Implement the training-time forward pass for batch normalization.   #
    # Use minibatch statistics to compute the mean and variance, use these      #
    # statistics to normalize the incoming data, and scale and shift the        #
    # normalized data using gamma and beta.                                     #
    #                                                                           #
    # You should store the output in the variable out. Any intermediates that   #
    # you need for the backward pass should be stored in the cache variable.    #
    #                                                                           #
    # You should also use your computed sample mean and variance together with  #
    # the momentum variable to update the running mean and running variance,    #
    # storing your result in the running_mean and running_var variables.        #
    #############################################################################
    pass
    sample_mean = np.mean(x, axis=0)
    x_minus_mean = x - sample_mean
    sq = x_minus_mean ** 2
    var = 1. / N * np.sum(sq, axis=0)
    sqrtvar = np.sqrt(var + eps)
    ivar = 1. / sqrtvar
    x_norm = x_minus_mean * ivar
    gammax = gamma * x_norm
    out = gammax + beta
    running_var = momentum * running_var + (1 - momentum) * var
    running_mean = momentum * running_mean + (1 - momentum) * sample_mean

    cache = (out, x_norm, beta, gamma, x_minus_mean, ivar, sqrtvar, var, eps)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  elif mode == 'test':
    #############################################################################
    # TODO: Implement the test-time forward pass for batch normalization. Use   #
    # the running mean and variance to normalize the incoming data, then scale  #
    # and shift the normalized data using gamma and beta. Store the result in   #
    # the out variable.                                                         #
    #############################################################################
    x = (x - running_mean) / np.sqrt(running_var)
    out = x * gamma + beta
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  return out, cache


def batchnorm_backward(dout, cache):
  """
  Backward pass for batch normalization.

  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.

  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.

  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
  dx, dgamma, dbeta = None, None, None
  N, D = dout.shape
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #############################################################################
  out, x_norm, beta, gamma, x_minus_mean, ivar, sqrtvar, var, eps=cache
  N = x_norm.shape[0]
  dout_=gamma*dout
  dvar=np.sum(dout_*x_minus_mean*-0.5*(var+eps)**-1.5,axis=0)
  di=dout_*ivar+dvar*2*x_minus_mean/N
  dmean=-1*np.sum(di,axis=0)/N
  dx=di+dmean
  dgamma=np.sum(dout*x_norm,axis=0)
  dbeta=np.sum(dout,axis=0)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.

  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  #############################################################################
  # TODO: Implement the forward pass for spatial batch normalization.         #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  N,C,H,W=x.shape
  x=np.transpose(x,(0,2,3,1))
  x=x.reshape(N*H*W,C)
  out,cache=batchnorm_forward(x, gamma, beta, bn_param)
  out=out.reshape(N,H,W,C).transpose(0,3,1,2)
  

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.

  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass

  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  #############################################################################
  # TODO: Implement the backward pass for spatial batch normalization.        #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  N,C,H,W=dout.shape
  dout=dout.transpose(0,2,3,1).reshape(N*H*W,C)
  dx,dgamma,dbeta=batchnorm_backward(dout, cache)
  
  dx=dx.reshape(N,H,W,C).transpose(0,3,1,2)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta

