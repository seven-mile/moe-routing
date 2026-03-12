import argparse
import json
import logging
import math
import os
import random
import torch
from pathlib import Path
from itertools import accumulate
from typing import List, Tuple, Dict, Any, Sequence, Optional

from lm_eval import evaluator

# --- BoTorch / GPyTorch ---
from botorch.models import SingleTaskGP, ModelListGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize
from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood


# --- Basic Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)

# ----------------------------
# Param conversion (variable dimensions)
# ----------------------------
def convert_point_to_formula(point: Sequence[float]) -> Tuple[float, ...]:
    """Converts a point (p0,p1,...,pn) to cumulative product formula."""
    return tuple(accumulate(point, lambda x, y: x * y))

def convert_formula_to_point(formula: Sequence[float]) -> Tuple[float, ...]:
    """Converts cumulative product formula back to (p0,p1,...,pn)."""
    if len(formula) == 0:
        return tuple()
    result = [formula[0]]
    for i in range(1, len(formula)):
        result.append(formula[i] / formula[i - 1])
    return tuple(result)

def is_monotone_nonincreasing(formula: Sequence[float]) -> bool:
    return all(formula[i] >= formula[i + 1] for i in range(len(formula) - 1))

# ----------------------------
# lm_eval evaluation (modified)
# ----------------------------
def evaluate_with_lmeval(args: argparse.Namespace, formula: Tuple[float, ...]) -> Tuple[float, float]:
    """
    Returns:
      score: gsm8k_cot exact_match,flexible-extract (higher is better)
      mean_topk: results["topk"]["mean_topk"] (lower is better)
    """
    logging.info(f"Setting up lm_eval for formula: {formula}")

    model_args_dict = {
        "model": args.model_name,
        "base_url": args.base_url,
        "num_concurrent": args.num_concurrent,
        "max_retries": args.max_retries,
        "tokenized_requests": False,
        "assisted_action": {
            "file": "configs/ppl_to_ks.py",
            "function": "spec_with_list_layer_range",
            "args": [
                list(formula),
                [20, 25],
            ],
        },
    }

    results = evaluator.simple_evaluate(
        model="local-completions",
        model_args=model_args_dict,
        gen_kwargs={
            "temperature": 0.0,
        },
        tasks=["gsm8k_cot"],
        limit=args.num_samples if args.num_samples > 0 else None,
    )

    # score
    try:
        score = results["results"]["gsm8k_cot"]["exact_match,flexible-extract"]
    except KeyError:
        logging.error("Missing gsm8k_cot exact_match,flexible-extract in lm_eval results.")
        score = 0.0

    # mean_topk
    mean_topk = None
    try:
        mean_topk = results["topk"]["mean_topk"]
    except Exception:
        # 有些 lm_eval 输出可能没有 topk 段
        logging.error("Missing topk.mean_topk in lm_eval results. Set mean_topk=inf as penalty.")
        mean_topk = float("inf")

    logging.info(f"lm_eval finished. score={score:.4f}, mean_topk={mean_topk}")
    return float(score), float(mean_topk)

# ----------------------------
# Pareto computation (minimize)
# ----------------------------
def pareto_mask_min(Y: torch.Tensor) -> torch.Tensor:
    """
    Y: (n, 2) minimization objectives: [1-score, mean_topk]
    returns: boolean mask of nondominated points (Pareto frontier)
    """
    n = Y.shape[0]
    is_pareto = torch.ones(n, dtype=torch.bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        # A point j dominates i (min) if: Y[j] <= Y[i] all dims AND < in at least one dim
        dominates_i = (Y <= Y[i]).all(dim=-1) & (Y < Y[i]).any(dim=-1)
        dominates_i[i] = False
        if dominates_i.any():
            is_pareto[i] = False
            continue
        # If i is pareto, it may dominate others
        dominated_by_i = (Y[i] <= Y).all(dim=-1) & (Y[i] < Y).any(dim=-1)
        dominated_by_i[i] = False
        is_pareto[dominated_by_i] = False
    return is_pareto

def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default

# ----------------------------
# Main optimization (BoTorch MOO)
# ----------------------------
def run_optimization(args: argparse.Namespace):
    # Search space bounds for point = (p0,p1,...,pn)
    # p0 in [1,20], p1,p2,...,pn in [0,1]
    n_dims = args.n_dims
    lower_bounds = [1.0] + [0.0] * (n_dims - 1)
    upper_bounds = [20.0] + [1.0] * (n_dims - 1)
    bounds = torch.tensor([
        lower_bounds,
        upper_bounds,
    ], dtype=torch.double)

    output_path = Path(args.output_file)
    results: List[Dict[str, Any]] = []
    if output_path.exists():
        try:
            with open(output_path, "r") as f:
                results = json.load(f)
            logging.info(f"Loaded {len(results)} existing results from {output_path}")
        except json.JSONDecodeError:
            logging.warning(f"Could not parse {output_path}. Starting fresh.")
            results = []

    # Helper: lookup evaluated formula
    evaluated = {}
    for r in results:
        fstr = r.get("formula")
        if fstr:
            evaluated[fstr] = r

    def point_to_formula_str(point: Sequence[float]) -> Tuple[Tuple[float, ...], str]:
        formula = convert_point_to_formula(point)
        return formula, str(tuple(float(x) for x in formula))

    # Build initial training data from existing results
    train_X_list = []
    train_Y_list = []

    for r in results:
        fstr = r.get("formula")
        if not fstr or fstr in ("baseline", "full"):
            continue
        try:
            formula = eval(fstr) if isinstance(fstr, str) else tuple(fstr)
            point = convert_formula_to_point(formula)
            # BoTorch will maximize [score, -mean_topk]
            score = safe_float(r.get("score"), 0.0)
            mean_topk = safe_float(r.get("mean_topk"), float("inf"))
            # If mean_topk is inf, still include but it will be bad
            train_X_list.append(point)
            train_Y_list.append([score, -mean_topk])
        except Exception:
            continue

    train_X = torch.tensor(train_X_list, dtype=torch.double) if train_X_list else torch.empty(0, n_dims, dtype=torch.double)
    train_Y = torch.tensor(train_Y_list, dtype=torch.double) if train_Y_list else torch.empty(0, 2, dtype=torch.double)

    # If not enough initial data, do random warmup evaluations
    def random_point() -> Tuple[float, ...]:
        p0 = random.uniform(1.0, 10.0)
        rest = [random.uniform(0.0, 1.0) for _ in range(n_dims - 1)]
        return tuple([p0] + rest)

    def evaluate_point(point: Tuple[float, ...]) -> Tuple[float, float, Tuple[float, ...], str]:
        formula, fstr = point_to_formula_str(point)

        # Keep original monotonicity check & penalty path (should always pass with p1..p3 in [0,1])
        if not is_monotone_nonincreasing(formula):
            logging.warning(f"Skipping invalid (non-monotonic) formula: {formula}")
            return 0.0, float("inf"), formula, fstr

        if fstr in evaluated:
            r = evaluated[fstr]
            score = safe_float(r.get("score"), 0.0)
            mean_topk = safe_float(r.get("mean_topk"), float("inf"))
            logging.info(f"Formula {formula} already evaluated. score={score:.4f}, mean_topk={mean_topk}. Skipping.")
            return score, mean_topk, formula, fstr

        score, mean_topk = evaluate_with_lmeval(args, formula)

        entry = {
            "formula": fstr,
            "score": score,
            "mean_topk": mean_topk,
            # Keep explicit minimization objectives for convenience
            "obj_1_minus_score": 1.0 - score,
            "obj_mean_topk": mean_topk,
        }
        results.append(entry)
        evaluated[fstr] = entry

        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)

        logging.info(f"Saved result. formula={formula}, score={score:.4f}, mean_topk={mean_topk}")
        return score, mean_topk, formula, fstr

    # Warm start if needed
    n_init_needed = max(0, args.n_initial_points - train_X.shape[0])
    if n_init_needed > 0:
        logging.info(f"Not enough initial data ({train_X.shape[0]}). Doing {n_init_needed} random initial evaluations.")
        for _ in range(n_init_needed):
            pt = random_point()
            score, mean_topk, _, fstr = evaluate_point(pt)
            # append to training
            x = torch.tensor([pt], dtype=torch.double)
            y = torch.tensor([[score, -mean_topk]], dtype=torch.double)
            train_X = torch.cat([train_X, x], dim=0) if train_X.numel() else x
            train_Y = torch.cat([train_Y, y], dim=0) if train_Y.numel() else y

    # Main BoTorch loop
    total_evals_target = args.n_calls
    while train_X.shape[0] < total_evals_target:
        # Normalize inputs to unit cube for GP stability
        train_Xn = normalize(train_X, bounds=bounds)

        # Build 2-output model as ModelListGP of two SingleTaskGPs
        model_1 = SingleTaskGP(train_Xn, train_Y[:, [0]])
        model_2 = SingleTaskGP(train_Xn, train_Y[:, [1]])
        model = ModelListGP(model_1, model_2)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        # Reference point for qNEHVI (in maximize space)
        # Make it slightly worse than the observed mins
        Y_obs = train_Y
        y_min = Y_obs.min(dim=0).values
        y_max = Y_obs.max(dim=0).values
        span = (y_max - y_min).clamp_min(1e-6)
        ref_point = (y_min - 0.1 * span).tolist()

        # Partitioning over observed points (maximize)
        # partitioning = NondominatedPartitioning(ref_point=torch.tensor(ref_point, dtype=torch.double), Y=Y_obs)

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([args.mc_samples]))

        acq = qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,
            X_baseline=train_Xn,
            sampler=sampler,
            prune_baseline=True,
            cache_root=True,
            # important: passing partitioning speeds things up
            # partitioning=partitioning,
        )

        # Optimize acquisition on unit cube
        candidate_n, _ = optimize_acqf(
            acq_function=acq,
            bounds=torch.tensor([[0.0] * n_dims, [1.0] * n_dims], dtype=torch.double),
            q=1,
            num_restarts=args.num_restarts,
            raw_samples=args.raw_samples,
            options={"batch_limit": 5, "maxiter": 200},
        )
        cand_n = candidate_n.detach().double()
        cand = unnormalize(cand_n, bounds=bounds).view(-1)

        point = tuple(float(cand[i]) for i in range(n_dims))
        formula, fstr = point_to_formula_str(point)

        # Dedup: if already evaluated, fallback to random new point
        tries = 0
        while fstr in evaluated and tries < 5:
            pt = random_point()
            formula, fstr = point_to_formula_str(pt)
            point = pt
            tries += 1

        score, mean_topk, _, _ = evaluate_point(point)

        # append new data
        x_new = torch.tensor([point], dtype=torch.double)
        y_new = torch.tensor([[score, -mean_topk]], dtype=torch.double)
        train_X = torch.cat([train_X, x_new], dim=0)
        train_Y = torch.cat([train_Y, y_new], dim=0)

        logging.info(f"Progress: {train_X.shape[0]}/{total_evals_target} evals done.")

    # Compute Pareto frontier in minimization space: [1-score, mean_topk]
    obj_min = []
    for r in results:
        if "score" in r and "mean_topk" in r:
            obj_min.append([1.0 - float(r["score"]), float(r["mean_topk"])])
    if len(obj_min) == 0:
        logging.warning("No valid results to compute Pareto frontier.")
        return

    Ymin = torch.tensor(obj_min, dtype=torch.double)
    mask = pareto_mask_min(Ymin)

    pareto_entries = []
    idx_map = 0
    for i, r in enumerate(results):
        if "score" not in r or "mean_topk" not in r:
            continue
        if mask[idx_map].item():
            pareto_entries.append(r)
        idx_map += 1

    # Sort Pareto by (1-score, mean_topk)
    pareto_entries_sorted = sorted(
        pareto_entries,
        key=lambda d: (float(d.get("obj_1_minus_score", 1.0 - float(d["score"]))),
                       float(d.get("obj_mean_topk", float(d["mean_topk"])))),
    )

    pareto_path = Path(args.pareto_output_file)
    with open(pareto_path, "w") as f:
        json.dump(pareto_entries_sorted, f, indent=4)

    logging.info(f"Pareto frontier saved to: {pareto_path}")
    logging.info("Top Pareto points (first 10):")
    for e in pareto_entries_sorted[:10]:
        logging.info(
            f"formula={e['formula']}, 1-score={e.get('obj_1_minus_score', 1.0 - e['score']):.4f}, mean_topk={e['mean_topk']}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-objective BO (BoTorch) to optimize speculative decoding with lm_eval.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- lm_eval / local-completions ---
    parser.add_argument("--base_url", type=str, default="http://127.0.0.1:8000/v1/completions")
    parser.add_argument("--num_concurrent", type=int, default=128)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--num_samples", type=int, default=-1, help="lm_eval limit; -1 for all")

    # --- Optimization ---
    parser.add_argument("--output_file", type=str, default="optimization_results_lmeval.json")
    parser.add_argument("--pareto_output_file", type=str, default="pareto_frontier.json")

    parser.add_argument("--n_dims", type=int, default=4, help="Number of dimensions for the search space (default: 4).")
    parser.add_argument("--n_calls", type=int, default=100, help="Total number of evaluations to perform (including existing).")
    parser.add_argument("--n_initial_points", type=int, default=10, help="Random initial evaluations if history is insufficient.")

    # --- BoTorch knobs ---
    parser.add_argument("--mc_samples", type=int, default=128, help="MC samples for qNEHVI.")
    parser.add_argument("--num_restarts", type=int, default=10, help="optimize_acqf restarts.")
    parser.add_argument("--raw_samples", type=int, default=256, help="optimize_acqf raw samples.")

    args = parser.parse_args()
    run_optimization(args)
