def xav_init(nd_arr_shape, n):
  # n should be the number of columns in the previous layer
  xav_arr = np.empty(nd_arr_shape)
  xav_arr = np.random.uniform(-1 / np.sqrt(n), 1 / np.sqrt(n), nd_arr_shape)
  return xav_arr
