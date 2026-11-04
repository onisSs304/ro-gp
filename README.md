## Riemannian Optimization for covariance matrix estimation in Gaussian process with categorical inputs

### Negative log-likelihood function

$$
\hat{\beta}, \hat{T} \in \arg \min _{\beta \in \mathbb{R}^m, T } \log (\det (ZTZ^T \circ K_X + \sigma ^2 Id)) + (Y - F\beta)^t (ZTZ^T \circ K_X + \sigma ^2 Id)^{-1} (Y - F\beta)
$$
We can simply write: $K(T) = ZTZ^T \circ K_X + \sigma ^2 Id$, $Z$ is the one-hot coding matrix depending on the categorical data, $T$ is the covariance matrix to be estimated, K_X is the covariance matrix corresponding to the continuous part. We can minimize this function in function of $\beta$ firstly, this gives $$
\hat{\beta} = (F^TK^{-1}F)^{-1}F^TK^{-1}y
$$

Now, our problem turns into optimization over a matrix space. Depending on the parametrization of $T$, we have another geometric structure: $T$ could be a SPD matrix in general or could be unit diagonal. 

Update 04/11/2026: 

Some ideas to deal with the geometry of full rank correlation matrices (SPD matrices with unit diagonal), denote $Cor^+(n)$: 

 - $Cor^+(m)$ as a embedded submanifold of $Sym^+(n)$; we use the inner product induced by the the ambient space (Affine invariant or Bures-Wasserstein), then $grad f(A) = Proj_A(grad \bar{f}(A))$, where $\bar{f}$ is the smooth extension of $f$ in $Sym^+(n)$.
 - $Cor^+(n)$ as a quotient manifold, $Cor^+(n) \cong Sym^+(n)/Diag^+(n)$, then try to the closed form for Riemannian metric, exponential map or Riemannian gradient as the instructions of the Boumal's book but it could be more computationally expensive (Actually the closed-form expression exists). 
 - $Cor^+(n) = Elliptone(n,n)$ in pymanopt directly (Elliptone(n,k) is rank-k, symmetric, semidefinite positive and unit diagonal matrices). However, in the implementation, they used the Frobenius inner product that could ignore the geometric properties of the space itself. 

Update 03/11/2026:

We implemented the Euclidean methods to optimize the problem, we choosen L-BFGS (already implemented in scipy package). For the gradient and hessian, to get a fair comparison with the package pymanopt, we used automatic differentiation implemented in the package autograd. 

Some empirical observations: 
 - For both SPD and SPD unit diagonal, if the Gaussian process is purely categorical, the resulting optimal matrix will be singular (minimum eigenvalue goes to 0)
 - In the presence of continuous part, we used RBF kernel function:
$$
k(x,x') = \exp (\frac{||x-x'||^2}{2l^2})
$$
 - We saw an interplay between lengthscale $l$ and the optimal matrix:
   + For SPD matrix in general, the bigger length-scale, the resulting matrix tends to the boundary of manifold SPD matrices. For smaller length-scales, the problem becomes more well-conditioned. Interestingly, if the length-scale is too small, l <= 0.005, the resulting matrix also becomes ill-conditioned. 
   + For SPD matrix with unit diagonal, the length-scale also affects the resulting matrix with the same way; to obtain the true covariance matrix, length-scale need not to be too large and too small.
 - From these observations, for ill-conditionned problems, we could use Bures-Wasserstein metric that is well-known favorable in this case (already proved for space of symmetric positive definite matrices but not verified for correlation matrices).  
