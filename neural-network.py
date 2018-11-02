import numpy as np

def backpropagation():
  """
  """
  # Something here
  return # something here

def pool_layer(X, pool_size=2):
  """
  The pooling layer of a neural network. This takes in an array of arbitrary
  shape. This function requires numpy to be
  imported as np:
  >>> import numpy as np
  Note: Currently, this only works if the last two layers are divisible by 2.
  This is an issue to be examined in the future. 

  Parameters
  ----------
  X : an ndarray from numpy.
  pool_size : The size of the square array to be pooled over the previous
  weights of the neural network. By default, this is set to 2.

  Returns
  -------
  pool_arr : an ndarray which collects the max value of the pool_size x
  pool_size array.

  Example
  -------
  >>> a = np.array([[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16]])
  >>> pool_layer(a, 2)
  array([[ 6.,  8.],
         [14., 16.]])
  
  """

  num_dims = len(X.shape)
  # initialization of the pool array by first initializating the shape of the
  # array.
  pool_shape = ()
  for i in range(num_dims - 2):
    pool_shape = pool_shape + (X.shape[i], )
  pool_shape = pool_shape + (X.shape[num_dims - 2] // 2, )
  pool_shape = pool_shape + (X.shape[num_dims - 1] // 2, )
  pool_arr = np.empty(pool_shape) # Initialization of the pool array
  # The parameters which np.max is taken over.
  axis_params = tuple(range(num_dims))[num_dims-2:num_dims]
  for i in range(pool_shape[num_dims - 2]):
    for j in range(pool_shape[num_dims - 1]):
      pool_arr[...,i,j] = np.max(X[...,2*i:2+2*i,2*j:2+2*j], axis=axis_params)
  return pool_arr

def normalization(X):
  A = np.empty_like(X)
  A = (X - np.mean(X)) / np.std(X)
  return A

def ADAM(learn_rate, beta1=0.9, beta2=0.999,
         func, theta, iterations, epsilon=10e-8):
  """
  This calculates the parameters theta which correspond to the local minimum of
  the function denoted by func. That is func(theta) will be a local minimum of
  the function given by func.

  Parameters
  ----------
  learn_rate : this is how much the gradient step of our function is weighted
  by.
  beta1 : An exponential decay rate for the moment estimate. It is in the
  interval [0, 1[.
  beta2 : An exponential decay rate for the moment estimate. It is in the
  interval [0, 1[.
  func : This is the loss function that is to be computed.
  theta : These are the initial parameter vector to our loss function.
  iterations : The number of times we wish to run the optimizer.
  epsilon : A parameter added so that vision by zero errors do not occur.
  
  Returns
  -------
  theta : an ndarray.

  Notes:
  ------
  This is the ADAM algorithm as outlined in 
  @article{DBLP:journals/corr/KingmaB14,
    author    = {Diedrik P. Kingma and
                 Jimmy Ba},
    title     = {Adam: {A} Method for Stocahstic Optimization},
    journal   = {CoRR},
    volume    = {abs/1412.6980},
    year      = {2014},
    url       = {http://arxiv.org/abs/1412.6980},
    archivePrefix = {arXiv},
    eprint    = {1412.6980},
    timestamp = {Mon, 13 Aug 2018 16:47:35 +0200},
    biburl    = {https://dblp.org/rec/bib/journals/corr/KingmaB14},
    bibsource = {dblp computer science bibliography, https://dblp.org}
  }
  
  This function requires numpy to be imported as np:
  >>> import numpy as np
  
  Further, it requires a compute gradient function to be called.
  """
  moment1 = np.zeros_like(theta) # initialize 1st moment vector
  moment2 = np.zeros_like(theta) # initialize 2nd moment vector
  t = 0 # initialize timestep
  for i in range(iterations):
    t += 1
    grad = compute_gradient(func, theta)
    moment1 = beta1 * moment1 + (1 - beta1) * grad 
    moment2 = beta2 * moment2 + (1 - beta2) * grad * grad 
    moment1hat = moment1 / (1 - beta1 ** t)
    moment2hat = moment2 / (1 - beta2 ** t)
    theta = theta - learn_rate * moment1hat / (np.sqrt(moment2hat) + epsilon)
  return theta

def ReLU(X):
  """
  Calculates the rectified linear unit component-wise for an ndarray. The
  rectified linear unit is defined component-wise as:
            -- x,   x >0 
  ReLU(x) = |
            -- 0, otherwise.
  
  Parameters
  ----------
  X : an ndarray.

  Returns
  -------
  out : ndarray
    Array with all negative elements set to 0 and positive elements left
    unchanged.

  Notes
  -----
  This function requires the module numpy to be imported as np. Include the
  line:
  >>> import numpy as np

  Examples
  --------
  >>> a = np.array([[1, -1], [-3, 2]])
  >>> ReLU(a)
  array([[1, 0],
         [0, 2]])
  """
  zeros = np.zeros_like(X)
  return np.maximum(zeros, X)

def ReLUgrad(X):
  grad = (X > 0)
  grad = grad.astype(float)
  return grad

def softmax(X):
  return np.exp(X) / np.sum(np.exp(X))

def softmax_grad(X):
  
def conv_layer(X, filter_arr, step_size=1):
  # Change so that it can take in larger sizes. I want to make it so that this
  # detects the number of color channels that the picture has.
  conv_array = np.empty_like(X)
  padded = pad(X, step_size)
  m = filter_arr.shape
  for i in range(X.shape[-2]):
    for j in range(X.shape[-1]):
      conv_array[...,i , j] = np.tensordot(padded[...,
                                                  i:m[-2]+i,
                                                  j:m[-1]+j],
                                                  filter_arr)
  return conv_array 

def conv_grad(X, filter_arr):
  #some code
  return #some code

def pad(X, pad_size=1):
  pad_array = X.copy()

  # Initializing the padding matrix
  zero_col_shape = ()
  for i in range(len(X.shape) - 1):
    zero_col_shape = zero_col_shape + (X.shape[i], )
  zero_col_shape = zero_col_shape + (1,) 
  zero_col = np.zeros(zero_col_shape)
  
  # Left and right append
  for i in range(pad_size):
    pad_array = np.c_[zero_col, pad_array, zero_col]
  
  zero_row_shape = ()
  n = len(pad_array.shape) - 1
  for i in range(n - 1):
    zero_row_shape = zero_row_shape + (pad_array.shape[i], )
  zero_row_shape = zero_row_shape + (1, )
  zero_row_shape = zero_row_shape + (pad_array.shape[n], )
  zero_row = np.zeros(zero_row_shape)

  # Top and bottom append
  for i in range(pad_size):	
    pad_array = np.r_['-2', zero_row, pad_array, zero_row]
  return pad_array

def create_filter(filter_size1=3, filter_size2=3):
  return np.random.randn(filter_size1, filter_size2)

def xav_init(nd_arr_shape, n):
  """
  This is the recommended initalization of Xavier Glorot and Yoshua Bengio. The
  bibtex reference is as follows:
  @INPROCEEDINGS{Glorot10understandingthe,
    author = {Xavier Glorot and Yoshua Bengio},
    title = {Understanding the difficulty of training deep feedforward neural
    networks},
    booktitle = {In Proceedings of the International Conference on Artifical
    Intelligence and Statistics (AISTATS'10). Society for Artificial
    Intelligence and Statistics},
    year = {2010}
  }
  """
  # n should be the number of columns in the previous layer
  xav_arr = np.empty(nd_arr_shape)
  xav_arr = np.random.uniform(-1 / np.sqrt(n), 1 / np.sqrt(n), nd_arr_shape)
  return xav_arr

  # Code for mini-batch
  # import time
  # from tqdm import tqdm
  # for i in tqdm(range(10)):
  #   code to run
  # This shows how much time and the estimated time the operations will take
