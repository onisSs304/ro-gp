import argparse
import csv
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = SRC_DIR / "data" / "dataset.npz"

import numpy as np
from gp import GP
from euclidean_methods import l_bfgs

def load_data(path):
  data = np.load(path)
  cats = np.asarray(data["cats"], dtype=int)
  conts = np.asarray(data["conts"], dtype=float)
  y = np.asarray(data["y"], dtype=float).reshape(-1)
  F = np.asarray(data["F"], dtype=float) if "F" in data else None
  return cats, conts, y, F

def run_experiments(args):
  cats, conts, y, F = load_data(args.data)
  m = args.m
  gp = GP(m=m, y=y, cats=cats, conts=conts, l=args.length_scale, F=F, continuous_part = not args.no_continuous_part, sigma=args.sigma)
  T0 = np.eye(m)
  results = []
  histories = []
  optimal_matrices = []
  for run in range(args.n_runs):
    seed = args.seed + run
    rng = np.random.default_rng(seed)
    start_time = time.perf_counter()

    result = l_bfgs(gp=gp, T0=T0, method="L-BFGS-B", n_starts=args.n_starts, gtol=args.gtol, ftol=args.ftol, maxcor=args.maxcor, record_history=True, rng=rng)

    elapsed = time.perf_counter() - start_time
    best = min(candidates, key=lambda r: r.fun)
    results.append({
            "run": run,
            "seed": seed,
            "nll": best.fun,
            "success": best.success,
            "iterations": best.nit,
            "evaluations": best.nfev,
            "time": elapsed,
            "message": best.message,
        })
    histories.append(best.history)
    optimal_matrices.append(best.T_opt)

    print(f"Run {run + 1:02d} | "
            f"NLL = {best.fun:.8f} | "
            f"success = {best.success} | "
            f"iterations = {best.nit} | "
            f"time = {elapsed:.2f}s")
  output_dir = Path(args.output)
  output_dir.mkdir(parents=True, exist_ok=True)
  with open(output_dir / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
  np.savez_compressed(
        output_dir / "optimization_results.npz",
        T_opt=np.stack(optimal_matrices),
        histories=np.array(histories, dtype=object),
        nll=np.array([r["nll"] for r in results]),
    )


  losses = np.array([r["nll"] for r in results])
  print("\n========== EXPERIMENT SUMMARY ==========")
  print(f"Number of runs: {args.n_runs}")
  print(f"Best NLL:       {losses.min():.8f}")
  print(f"Mean NLL:       {losses.mean():.8f}")
  print(f"Std NLL:        {losses.std():.8f}")
  print(f"Worst NLL:      {losses.max():.8f}")
  print(f"Results saved in: {output_dir.resolve()}")

if __name__  == "main":
  parser = argparse.ArgumentParser()
  parser.add_argument("--data", type=str, default=str(DEFAULT_DATA))
  parser.add_argument("--m", type=int, required=True)

  parser.add_argument("--length-scale", type=float, default=1.0)
  parser.add_argument("--sigma", type=float, default=1e-3)
  parser.add_argument("--no-continuous-part", action="store_true")

  parser.add_argument("--n-runs", type=int, default=10)
  parser.add_argument("--n-starts", type=int, default=5)
  parser.add_argument("--gtol", type=float, default=1e-6)
  parser.add_argument("--ftol", type=float, default=1e-12)
  parser.add_argument("--maxcor", type=int, default=10)

  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--output", type=str, default=str(SRC_DIR/"result"/"l_bfgs"))

  args = parser.parse_args()

  run_experiments(args)
    

    

