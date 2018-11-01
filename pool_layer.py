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
	# initialization of pool array by first initialization the shape of the
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
