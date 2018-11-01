def normalization(X):
  A = np.empty_like(X)
  A = (X - np.mean(X)) / np.std(X)
  return A
