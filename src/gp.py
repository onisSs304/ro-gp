import autograd.numpy as anp
from autograd.scipy.linalg import solve_triangular

class GP: 
  def __init__ (self, m, y, cats, conts, l=1, F=None, continuous_part=True, sigma=1e-3):
    y = anp.asarray(y, float)
    self.y = y
    self.n = self.y.shape[0] # number of observations
    self.l = l # lengthscale of continuous inputs Gaussian process; in scikit-learn, l is 1 by default
    self.m = m # number of category levels
    self.sigma = sigma
    self.conts = conts
    if F is not None:
      self.F = anp.asarray(F, float) 
    else:
      self.F = None
    if continuous_part:
      self.T_cont =self._RBF_kernel()  
    else:
      self.T_cont = anp.ones((self.n,self.n)) 
    self.Z = anp.eye(m)[cats] # one-hot matrix (n x m)
    self._cholesky, self._kin = None, None

  def _RBF_kernel(self): # RBF kernel of Gaussian processes, K(x,x')=exp(-||x-x'||^2/(2l^2)), l is lengthscale
    X = anp.asarray(self.conts)
    D = (X[:, None] - X[None, :])**2
    return anp.exp(-D / (2 * self.l**2))

  def _K(self, T):
    return (self.Z @ T @ self.Z.T) * self.T_cont + self.sigma * anp.eye(self.n)

  def _chol(self, T):
    return anp.linalg.cholesky(self._K(T))

  def _solve(self, L, B):
    Y = solve_triangular(L, B, lower=True)
    return solve_triangular(L.T, Y, lower=False)

  def _kinv(self, T):
    L = self._chol(T)
    I = anp.eye(self.n)
    Y = solve_triangular(L, I, lower=True)
    return solve_triangular(L.T, Y, lower=False)

  def beta_hat(self, T):
    if self.F is None:
        return 0
    L = self._chol(T)
    K1F = self._solve(L, self.F)
    K1y = self._solve(L, self.y)

    return anp.linalg.solve(
        self.F.T @ K1F,
        self.F.T @ K1y
    )
  
  def nll(self, T):
    L = self._chol(T)
    if self.F is not None:
        residual = self.y - self.F @ self.beta_hat(T)
    else:
        residual = self.y
    A = self._solve(L, residual)
    return ( 0.5 * residual.T @ A + anp.sum(anp.log(anp.diag(L)))+ self.n / 2 * anp.log(2 * anp.pi) )
  
    
