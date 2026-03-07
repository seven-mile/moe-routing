import argparse
import json
import logging
import math
import random
from pathlib import Path
from typing import List, Dict, Any, Sequence, Tuple

from lm_eval import evaluator

import torch

# --- BoTorch / GPyTorch ---
from botorch.models import SingleTaskGP, ModelListGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)


# --------------------------
# Utilities: t <-> p
# --------------------------
def formula_to_point(formula: Sequence[float], eps: float = 1e-12) -> Tuple[float, float, float, float]:
    """(t0,t1,t2,t3) -> (p0,p1,p2,p3) with safeguards."""
    t0, t1, t2, t3 = map(float, formula)
    p0 = t0
    p1 = t1 / max(t0, eps)
    p2 = t2 / max(t1, eps)
    p3 = t3 / max(t2, eps)
    return (p0, p1, p2, p3)

def point_to_formula(point: Sequence[float]) -> Tuple[float, float, float, float]:
    """(p0,p1,p2,p3) -> (t0,t1,t2,t3) via cumulative product."""
    p0, p1, p2, p3 = map(float, point)
    t0 = p0
    t1 = t0 * p1
    t2 = t1 * p2
    t3 = t2 * p3
    return (t0, t1, t2, t3)

def clamp_point(
    p: Sequence[float],
    p0_min: float = 1.0,
    p0_max: float = 10.0,
    eps_p: float = 1e-4,
) -> Tuple[float, float, float, float]:
    p0, p1, p2, p3 = map(float, p)
    p0 = min(max(p0, p0_min), p0_max)
    # NOTE: keep p1..p3 in (0,1]
    p1 = min(max(p1, eps_p), 1.0)
    p2 = min(max(p2, eps_p), 1.0)
    p3 = min(max(p3, eps_p), 1.0)
    return (p0, p1, p2, p3)


# --------------------------
# Partition (48-layer default)
# --------------------------
def default_partition(num_layers: int) -> Tuple[slice, slice, slice, slice]:
    """
    A: [0, 24)
    B: [24, 30)
    C: [30, 36)
    D: [36, num_layers)
    If num_layers != 48, scale from the 48-template.
    """
    if num_layers == 48:
        a1, b1, c1 = 24, 30, 36
    else:
        def s(x: int) -> int:
            return int(round(x / 48.0 * num_layers))
        a1 = max(1, min(num_layers, s(24)))
        b1 = max(a1, min(num_layers, s(30)))
        c1 = max(b1, min(num_layers, s(36)))

    A = slice(0, a1)
    B = slice(a1, b1)
    C = slice(b1, c1)
    D = slice(c1, num_layers)
    return A, B, C, D


# --------------------------
# Build layer_cfgs (t-space)
# --------------------------
def apply_deltas_to_point(base_p: Tuple[float, float, float, float],
                          deltas: Sequence[float],
                          dim: int,
                          segment: str) -> Tuple[float, float, float, float]:
    """
    dim=6: deltas = [dB1,dB2,dB3, dC1,dC2,dC3]  (log-space)
    dim=8: deltas = [dB0,dB1,dB2,dB3, dC0,dC1,dC2,dC3]
    segment in {"B","C"}
    """
    p0, p1, p2, p3 = base_p
    if dim == 6:
        if segment == "B":
            d1, d2, d3 = deltas[0:3]
        elif segment == "C":
            d1, d2, d3 = deltas[3:6]
        else:
            raise ValueError(segment)
        # p0 fixed
        p1 *= math.exp(float(d1))
        p2 *= math.exp(float(d2))
        p3 *= math.exp(float(d3))
        return (p0, p1, p2, p3)

    if dim == 8:
        if segment == "B":
            d0, d1, d2, d3 = deltas[0:4]
        elif segment == "C":
            d0, d1, d2, d3 = deltas[4:8]
        else:
            raise ValueError(segment)
        p0 *= math.exp(float(d0))
        p1 *= math.exp(float(d1))
        p2 *= math.exp(float(d2))
        p3 *= math.exp(float(d3))
        return (p0, p1, p2, p3)

    raise ValueError(f"dim must be 6 or 8, got {dim}")

def build_layer_cfgs_tspace(
    base_formula: Sequence[float],
    log_deltas: Sequence[float],
    dim: int,
    num_layers: int,
    *,
    p0_min: float = 1.0,
    p0_max: float = 10.0,
    eps_p: float = 1e-4,
) -> List[List[float]]:
    """
    Returns layer_cfgs: list length num_layers, each [t0,t1,t2,t3]
    A & D use base_formula; B and C use p-space log_deltas around base.
    """
    base_formula = tuple(float(x) for x in base_formula)
    base_p = clamp_point(formula_to_point(base_formula), p0_min=p0_min, p0_max=p0_max, eps_p=eps_p)

    A, B, C, D = default_partition(num_layers)

    # Segment formulas in t-space
    # A/D: base
    fA = base_formula
    fD = base_formula

    # B/C: apply deltas in p-space, then convert back to t-space
    pB = clamp_point(apply_deltas_to_point(base_p, log_deltas, dim, "B"), p0_min=p0_min, p0_max=p0_max, eps_p=eps_p)
    pC = clamp_point(apply_deltas_to_point(base_p, log_deltas, dim, "C"), p0_min=p0_min, p0_max=p0_max, eps_p=eps_p)
    fB = point_to_formula(pB)
    fC = point_to_formula(pC)

    layer_cfgs: List[List[float]] = []
    for i in range(num_layers):
        if B.start <= i < B.stop:
            layer_cfgs.append([float(x) for x in fB])
        elif C.start <= i < C.stop:
            layer_cfgs.append([float(x) for x in fC])
        elif D.start <= i < D.stop:
            layer_cfgs.append([float(x) for x in fD])
        else:
            layer_cfgs.append([float(x) for x in fA])

    return layer_cfgs


# --------------------------
# lm_eval evaluation
# --------------------------
def evaluate_with_lmeval(args: argparse.Namespace, log_deltas: Sequence[float]) -> Tuple[float, float]:
    # IMPORTANT: num_layers unknown here, but config inside model will be known at runtime.
    # However assisted_action args are fixed before call.
    # Since your target is 48-layer, we generate for args.num_layers (default 48).
    layer_cfgs = build_layer_cfgs_tspace(
        base_formula=args.base_formula,
        log_deltas=log_deltas,
        dim=args.dim,
        num_layers=args.num_layers,
        p0_min=args.p0_min,
        p0_max=args.p0_max,
        eps_p=args.eps_p,
    )

    model_args_dict = {
        "model": args.model_name,
        "base_url": args.base_url,
        "num_concurrent": args.num_concurrent,
        "max_retries": args.max_retries,
        "tokenized_requests": False,
        "assisted_action": {
            "file": "configs/ppl_to_ks.py",
            "function": "spec_from_layer_cfgs",
            "args": [
                layer_cfgs,  # fixed via partial semantics
            ],
            "kwargs": {},  # in case your loader supports kwargs (you said it does)
        },
    }

    results = evaluator.simple_evaluate(
        model="local-completions",
        model_args=model_args_dict,
        tasks=["gsm8k_cot"],
        limit=args.num_samples if args.num_samples > 0 else None,
    )

    try:
        score = float(results["results"]["gsm8k_cot"]["exact_match,flexible-extract"])
    except Exception:
        logging.error("Missing gsm8k_cot exact_match,flexible-extract; set score=0.")
        score = 0.0

    try:
        mean_topk = float(results["topk"]["mean_topk"])
    except Exception:
        logging.error("Missing topk.mean_topk; set mean_topk=inf.")
        mean_topk = float("inf")

    logging.info(f"lm_eval done. score={score:.4f}, mean_topk={mean_topk}")
    return score, mean_topk


# --------------------------
# Pareto helper
# --------------------------
def pareto_mask_min(Y: torch.Tensor) -> torch.Tensor:
    """Y: (n,2) minimize objectives [1-score, mean_topk]."""
    n = Y.shape[0]
    is_pareto = torch.ones(n, dtype=torch.bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        dominates_i = (Y <= Y[i]).all(dim=-1) & (Y < Y[i]).any(dim=-1)
        dominates_i[i] = False
        if dominates_i.any():
            is_pareto[i] = False
            continue
        dominated_by_i = (Y[i] <= Y).all(dim=-1) & (Y[i] < Y).any(dim=-1)
        dominated_by_i[i] = False
        is_pareto[dominated_by_i] = False
    return is_pareto


# --------------------------
# Main optimization
# --------------------------
def run_optimization(args: argparse.Namespace):
    output_path = Path(args.output_file)
    pareto_path = Path(args.pareto_output_file)

    dim = int(args.dim)

    # Search bounds in log-space: each delta in [-ln(R), ln(R)]
    lo = -math.log(args.delta_ratio)
    hi = math.log(args.delta_ratio)
    bounds = torch.tensor([[lo] * dim, [hi] * dim], dtype=torch.double)

    results: List[Dict[str, Any]] = []
    evaluated: Dict[str, Dict[str, Any]] = {}

    if output_path.exists():
        try:
            results = json.loads(output_path.read_text())
            logging.info(f"Loaded {len(results)} existing results from {output_path}")
        except Exception:
            logging.warning(f"Could not parse {output_path}; starting fresh.")
            results = []

    for r in results:
        key = r.get("key")
        if key:
            evaluated[key] = r

    def key_of(log_deltas: Sequence[float]) -> str:
        rd = [round(float(x), 8) for x in log_deltas]
        return f"base={tuple(args.base_formula)}|dim={dim}|R={args.delta_ratio}|d={tuple(rd)}"

    def eval_once(log_deltas: Sequence[float]) -> Tuple[float, float]:
        key = key_of(log_deltas)
        if key in evaluated:
            r = evaluated[key]
            return float(r["score"]), float(r["mean_topk"])

        score, mean_topk = evaluate_with_lmeval(args, log_deltas)

        entry = {
            "key": key,
            "base_formula": list(args.base_formula),
            "dim": dim,
            "delta_ratio": args.delta_ratio,
            "log_deltas": list(map(float, log_deltas)),
            "score": score,
            "mean_topk": mean_topk,
            "obj_1_minus_score": 1.0 - score,
            "obj_mean_topk": mean_topk,
            "num_layers": args.num_layers,
        }
        results.append(entry)
        evaluated[key] = entry
        output_path.write_text(json.dumps(results, indent=2))
        return score, mean_topk

    # Warm-start
    train_X_list, train_Y_list = [], []
    for r in results:
        if r.get("dim") != dim:
            continue
        if tuple(r.get("base_formula", [])) != tuple(args.base_formula):
            continue
        d = r.get("log_deltas")
        if not isinstance(d, list) or len(d) != dim:
            continue
        score = float(r.get("score", 0.0))
        mean_topk = float(r.get("mean_topk", float("inf")))
        train_X_list.append(d)
        train_Y_list.append([score, -mean_topk])  # maximize space

    train_X = torch.tensor(train_X_list, dtype=torch.double) if train_X_list else torch.empty(0, dim, dtype=torch.double)
    train_Y = torch.tensor(train_Y_list, dtype=torch.double) if train_Y_list else torch.empty(0, 2, dtype=torch.double)

    # Random initial if insufficient
    n_init_needed = max(0, args.n_initial_points - train_X.shape[0])
    if n_init_needed > 0:
        logging.info(f"Need {n_init_needed} random initial evals.")
        for _ in range(n_init_needed):
            d = [random.uniform(lo, hi) for _ in range(dim)]
            score, mean_topk = eval_once(d)
            x = torch.tensor([d], dtype=torch.double)
            y = torch.tensor([[score, -mean_topk]], dtype=torch.double)
            train_X = torch.cat([train_X, x], dim=0) if train_X.numel() else x
            train_Y = torch.cat([train_Y, y], dim=0) if train_Y.numel() else y

    # BO loop
    total_evals_target = args.n_calls
    while train_X.shape[0] < total_evals_target:
        Xn = normalize(train_X, bounds=bounds)

        model_1 = SingleTaskGP(Xn, train_Y[:, [0]])
        model_2 = SingleTaskGP(Xn, train_Y[:, [1]])
        model = ModelListGP(model_1, model_2)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        # ref point slightly worse than observed mins in maximize space
        Y_obs = train_Y
        y_min = Y_obs.min(dim=0).values
        y_max = Y_obs.max(dim=0).values
        span = (y_max - y_min).clamp_min(1e-6)
        ref_point = (y_min - 0.1 * span).tolist()

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([args.mc_samples]))
        acq = qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,
            X_baseline=Xn,
            sampler=sampler,
            prune_baseline=True,
            cache_root=True,
        )

        cand_n, _ = optimize_acqf(
            acq_function=acq,
            bounds=torch.tensor([[0.0] * dim, [1.0] * dim], dtype=torch.double),
            q=1,
            num_restarts=args.num_restarts,
            raw_samples=args.raw_samples,
            options={"batch_limit": 5, "maxiter": 200},
        )
        cand = unnormalize(cand_n.detach().double(), bounds=bounds).view(-1).tolist()

        # Dedup fallback
        tries = 0
        while key_of(cand) in evaluated and tries < 5:
            cand = [random.uniform(lo, hi) for _ in range(dim)]
            tries += 1

        score, mean_topk = eval_once(cand)
        x_new = torch.tensor([cand], dtype=torch.double)
        y_new = torch.tensor([[score, -mean_topk]], dtype=torch.double)
        train_X = torch.cat([train_X, x_new], dim=0)
        train_Y = torch.cat([train_Y, y_new], dim=0)

        logging.info(f"Progress: {train_X.shape[0]}/{total_evals_target}")

    # Pareto in minimization space: (1-score, mean_topk)
    rows = [r for r in results if tuple(r.get("base_formula", [])) == tuple(args.base_formula) and r.get("dim") == dim]
    if not rows:
        logging.warning("No rows to compute Pareto.")
        return

    Ymin = torch.tensor([[float(r["obj_1_minus_score"]), float(r["obj_mean_topk"])] for r in rows], dtype=torch.double)
    mask = pareto_mask_min(Ymin)

    pareto_rows = [rows[i] for i in range(len(rows)) if mask[i].item()]
    pareto_rows = sorted(pareto_rows, key=lambda r: (float(r["obj_1_minus_score"]), float(r["obj_mean_topk"])))
    pareto_path.write_text(json.dumps(pareto_rows, indent=2))
    logging.info(f"Pareto frontier saved to {pareto_path} (count={len(pareto_rows)})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # lm_eval / local-completions
    p.add_argument("--base_url", type=str, default="http://0.0.0.0:7000/v1/completions")
    p.add_argument("--num_concurrent", type=int, default=32)
    p.add_argument("--max_retries", type=int, default=3)
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-30B-A3B")
    p.add_argument("--num_samples", type=int, default=-1, help="lm_eval limit; -1 for full")

    # Base strategy (t-space)
    p.add_argument("--base_formula", type=float, nargs=4, required=True, help="Base (t0,t1,t2,t3) from Pareto set.")

    # Layer count assumption for building layer_cfgs
    p.add_argument("--num_layers", type=int, default=48)

    # Search dim: 6 or 8
    p.add_argument("--dim", type=int, choices=[6, 8], default=6)

    # log-delta range
    p.add_argument("--delta_ratio", type=float, default=3.0, help="Each p is scaled by exp(delta) with delta in [-ln(R), ln(R)].")

    # Clamp params (kept in optimizer, not ppl_to_ks)
    p.add_argument("--p0_min", type=float, default=1.0)
    p.add_argument("--p0_max", type=float, default=10.0)
    p.add_argument("--eps_p", type=float, default=1e-4)

    # Optimization I/O
    p.add_argument("--output_file", type=str, default="optimization_results_layerwise.json")
    p.add_argument("--pareto_output_file", type=str, default="pareto_frontier_layerwise.json")

    # BO budget
    p.add_argument("--n_calls", type=int, default=100)
    p.add_argument("--n_initial_points", type=int, default=10)

    # BoTorch knobs
    p.add_argument("--mc_samples", type=int, default=128)
    p.add_argument("--num_restarts", type=int, default=10)
    p.add_argument("--raw_samples", type=int, default=256)

    args = p.parse_args()
    run_optimization(args)
