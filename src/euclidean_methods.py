import numpy as np
import autograd.numpy as anp
from autograd import value_and_grad
from scipy.optimize import minimize

def l_bfgs(gp, T0, method="L-BFGS-B", n_starts=1, gtol=1e-6, ftol=1e-15, record_history=True, rng=None, maxcor=10):
  m = gp.m
  il = np.tril_indices(m)
  dg = np.diag_indices(m)

  S = np.zeros((m * m, len(il[0])))
  S[il[0] * m + il[1], np.arange(len(il[0]))] = 1

  def random_spd(m, rng, cond=10.0):
    Q, _ = np.linalg.qr(rng.standard_normal((m, m)))
    w = np.exp(rng.uniform(0, np.log(cond), m))
    return (Q * w) @ Q.T
  
  def build_L(p): # p is a vector that contains optimization parameters, L is the Cholesky factor of another matrix
    L = anp.reshape(anp.dot(S, p), (m, m))
    return anp.tril(L, -1) + anp.diag(anp.exp(anp.diag(L)))

  def unpack(p): # transform p to the covariance matrix
    L = build_L(p); return L @ L.T 

  def pack(T):
    L = np.linalg.cholesky(T); L[dg] = np.log(L[dg])
    return L[il]

  def objective(p):
    L = build_L(p); T = anp.dot(L, L.T); return gp.nll(T)

  fun_jac = value_and_grad(objective) # automatic differentiation via autograd
  rng = np.random.default_rng(0) if rng is None else rng
  starts = [pack(T0)] + [pack(random_spd(m, rng)) for _ in range(n_starts - 1)]
  results = []
  for p0 in starts:
    hist = [fun_jac(p0)[0]]
    opts = {}
    if method == "L-BFGS-B":
      opts = dict(maxiter=5000, maxfun=50000, ftol=ftol, maxcor=maxcor)
    if gtol is not None:
      opts["gtol"] = gtol
    r = minimize(fun_jac, p0, jac=True, method=method, options=opts, callback = (
            lambda xk: hist.append(fun_jac(xk)[0])
        ) if record_history else None)
    r.T_opt = unpack(r.x)
    r.history = np.array(hist)
    results.append(r)
  return results

import autograd.numpy as anp

def angles_to_L(a, m):
  rows = []
  k = 0
  for i in range(m):
    if i == 0:
      rows.append(anp.array([1.0]))
      continue
    ang = a[k:k+i]
    sn = anp.sin(ang)
    cs = anp.cos(ang)

    row = []
    for j in range(i):
      if j == 0:
        prefix = 1.0
      else:
        prefix = anp.prod(sn[:j])

      row.append(prefix * cs[j])
    row.append(anp.prod(sn))
    rows.append(anp.stack(row))
    k += i

    # Pad each row with zeros to obtain a square matrix
  L = anp.stack([anp.concatenate([row, anp.zeros(m - len(row))])for row in rows])

  return L

def angles_lbfgs(gp, method="L-BFGS-B", n_starts=1, gtol=1e-6, ftol=1e-, record_history=True, rng=None, maxcor=10):
  m = gp.m
  na = m * (m - 1) // 2
  def objective(a):
    L = angles_to_L(a, m)
    C = anp.dot(L, L.T)
    return gp.nll(C)
  fun_jac = value_and_grad(objective)
  a0 = np.full(na, np.pi / 2)
  eps = 1e-8
  bounds = [(eps, np.pi - eps)] * na
  rng = np.random.default_rng(0) if rng is None else rng
  starts = [a0] + [rng.uniform(eps, np.pi - eps, na) for _ in range(n_starts - 1)]
  results = []
  for a0 in starts:
    hist = [fun_jac(a0)[0]]
    opts = {}
    if method == "L-BFGS-B":
      opts = dict(maxiter=5000, maxfun=50000, ftol=ftol, maxcor=maxcor)
    if gtol is not None:
      opts["gtol"] = gtol
    r = minimize(fun_jac, a0, jac=True, method=method, bounds=bounds, options=opts, callback = (
            lambda xk: hist.append(fun_jac(xk)[0])
        ) if record_history else None)
    L_opt = angles_to_L(r.x, m)
    r.T_opt = anp.dot(L_opt, L_opt.T)
    r.history = np.array(hist)
    results.append(r)

  return results


  

  
  
    
