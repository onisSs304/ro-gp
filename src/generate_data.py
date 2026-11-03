
import numpy as np
from pathlib import Path


def generate_dataset(n=200, m=3, length_scale=0.2, sigma=1e-3, seed=42, L=5):
    rng = np.random.default_rng(seed)

    cats = rng.integers(0, m, size=n)

    conts = rng.uniform(0, L, size=n)

    T_true = np.array([
    [2.0, 0.6, 0.2],
    [0.6, 1.5, 0.4],
    [0.2, 0.4, 0.8],
    ])
    Z = np.eye(m)[cats]
    D = (conts[:, None] - conts[None, :])**2

    K_cont = np.exp(
        -D / (2 * length_scale**2)
    )

    # 6. Full covariance matrix
    K = (Z @ T_true @ Z.T) * K_cont
    K += sigma * np.eye(n)

    # 7. Generate observations
    y = rng.multivariate_normal(
        mean=np.zeros(n),
        cov=K,
    )

    # 8. Save dataset
    SRC_DIR = Path(__file__).resolve().parent

    output_dir = SRC_DIR / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_dir / "dataset.npz",
        cats=cats,
        conts=conts,
        y=y,
        T_true=T_true,
    )

    print("Dataset generated successfully!")
    print("Number of observations:", n)
    print("Number of categories:", m)
    print("T_true:\n", T_true)

if __name__ == "__main__":
    generate_dataset()

