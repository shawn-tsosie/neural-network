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
