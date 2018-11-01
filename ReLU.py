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
