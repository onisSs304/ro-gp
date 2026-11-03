### Riemannian Optimization for covariance matrix estimation in Gaussian process with categorical inputs

## Negative log-likelihood function

$$
\hat{\beta}, \hat{K} \in \arg \min _{\beta \in \mathbb{R}^m, K \in \mathrm{Sym}^+(L)} \log (\det (HKH^T \circ K_X + \sigma ^2 Id)) + (Y - F\beta)^t (HKH^T \circ K_X + \sigma ^2 Id)^{-1} (Y - F\beta)
$$
