import numpy as np
from scipy.linalg import cho_factor, cho_solve

class GP: 
  def __init__ (self, m, y, cats, conts, l=1, F=None, continuous_part=True, sigma=1e-3):
    y = np.asarray(y, float)
    self.y = y
    self.n = self.y.shape[0] # number of observations
    self.l = l # lengthscale of continuous inputs Gaussian process; in scikit-learn, l is 1 by default
    self.m = m # number of category levels
    self.sigma = sigma
    self.conts = conts
    if F is not None:
      self.F = np.asarray(F, float) 
    else:
      self.F = None
    if continuous_part:
      self.T_cont =self._RBF_kernel()  
    else:
      self.T_cont = np.ones((self.n,self.n)) 
    self.Z = np.eye(m)[cats] # one-hot matrix (n x m)
    self._cholesky, self._kin = None, None

  def _RBF_kernel(self): # RBF kernel of Gaussian processes, K(x,x')=exp(-||x-x'||^2/(2l^2)), l is lengthscale
    X = np.asarray(self.conts)
    D = (X[:, None] - X[None, :])**2
    return np.exp(-D / (2 * self.l**2))

  def _K(self, T):
    return (self.Z @ T @ self.Z.T) * self.T_cont + self.sigma * np.eye(self.n)

  def _chol(self, T):
    if self._cholesky is not None and np.array_equal(self._cholesky[0], T):
      return self._cholesky[1]
    c = cho_factor(self._K(T), lower=True)
    self._cholesky, self._kin = (T.copy(), c), None
    return c

  def _kinv(self, T):
    c = self._chol(T)
    if self._kin is None:
      self._kin = cho_solve(c, np.eye(self.n))
    return self._kin

  def beta_hat(self, T):
    if self.F is None:
        return 0
    c = self._chol(T)
    K1F = cho_solve(c, self.F)
    K1y = cho_solve(c, self.y)
    return linalg.solve(
        self.F.T @ K1F,
        self.F.T @ K1y
    )
  
  def nll(self, T): # Negative log-likelihood 
    if self.F is not None:
      residual = self.y - self.F @ self.beta_hat(T)
    else:
      residual = self.y
    c = self._chol(T)
    A = cho_solve(c, residual)
    return 1/2 * residual.T @ A + np.sum(np.log(np.diag(c[0]))) + self.n/2 * np.log(2*np.pi)
  
    
