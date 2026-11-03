import numpy as np
from autograd import value_and_grad

from gp import GP

# Load dataset
data = np.load("src/data/dataset.npz")

cats = data["cats"]
conts = data["conts"]
y = data["y"]

m = 3

print("Dataset shapes:")
print("cats:", cats.shape)
print("conts:", conts.shape)
print("y:", y.shape)

# Construct GP
gp = GP(
    m=m,
    y=y,
    cats=cats,
    conts=conts,
    l=0.2,
    continuous_part=True,
    sigma=1e-3,
)

print("\nGP constructed successfully.")

# Test covariance matrix
T = np.array([
    [2.0, 0.6, 0.2],
    [0.6, 1.5, 0.4],
    [0.2, 0.4, 0.8],
])

K = gp._K(T)

print("\nEigenvalues of K:")
print(np.linalg.eigvalsh(K))

# Test negative log-likelihood
print("\nTesting NLL...")
loss = gp.nll(T)
print("NLL:", loss)

# Test automatic differentiation
print("\nTesting Autograd...")
grad_fn = value_and_grad(lambda T: gp.nll(T))

loss, grad = grad_fn(T)

print("NLL:", loss)
print("Gradient:")
print(grad)

print("\nAll tests passed!")
