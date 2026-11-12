Consider the problem: $$\min _{\theta \in [0,2\pi]} sin(\theta)$$

The global minimum of this function is attained at $\theta=\frac{3\pi}{2}$ and $\sin (\theta)=-1$. Numerically, we often use gradient descent to get there, however this problem admits a local minimum at $\theta=0$. So this algorithm depends strictly on the initialization $\theta_0$. Indeed, if the initial point fall into $[0, \frac{\pi}{2})$, the GD will converge to $0$ and get stuck there. 

We relate our original problem to an optimization problem over the unit sphere in $\mathbb{R}^2$, that is minimizing the function $f(x)=Proj_y(x)$, where $Proj_y(x)$ is the orthogonal projection from $x$ to $y$-axis. This function is smooth and bounded from below and above.

We use the first order condition optimality and compute $grad f(x) = 0$ where $grad f(x) = (I_2 - xx^T)(0, 1)^T$ which gives us 2 critical points: $(0, -1)$ and $(0,1)$. 

Then continue using second-order condition optimality to deduce that $(0,-1)$ is the local minimizer of our problem, which is also the global minimizer for the problem (the sphere is a compact manifold without boundary, so f smooth on a compact manifold admits 1 global minimum and global minimum is also a local minimum). 
