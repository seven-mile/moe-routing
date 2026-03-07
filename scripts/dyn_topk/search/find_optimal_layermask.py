#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union, Optional

from lm_eval import evaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)


def _stable_json(obj: Any) -> str:
    """Stable JSON string for hashing / keys."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _make_key(formula: Tuple[float, ...], layer_mask: Any) -> str:
    payload = {"formula": list(formula), "layer_mask": layer_mask}
    h = hashlib.sha1(_stable_json(payload).encode("utf-8")).hexdigest()[:16]
    return h


def parse_formula(x: Any) -> Tuple[float, ...]:
    """
    Accepts:
      - string like "(10.0, 6.58, 1.275, 1.0)"
      - list/tuple of numbers
    Returns: tuple(float,...)
    """
    if isinstance(x, (list, tuple)):
        return tuple(float(v) for v in x)
    if isinstance(x, str):
        s = x.strip()
        # allow "baseline"/"full" or other markers, but those shouldn't be evaluated
        if s in ("baseline", "full"):
            raise ValueError(f"Non-evaluable formula tag: {s}")
        try:
            v = eval(s, {"__builtins__": {}})  # safe-ish: no builtins
        except Exception as e:
            raise ValueError(f"Failed to eval formula string: {x}") from e
        if not isinstance(v, (list, tuple)):
            raise ValueError(f"Formula string did not eval to list/tuple: {x}")
        return tuple(float(t) for t in v)
    raise TypeError(f"Unsupported formula type: {type(x)}")


def load_pareto_formulas(pareto_path: Path) -> List[Tuple[float, ...]]:
    data = json.loads(pareto_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("pareto_frontier.json must be a JSON list of entries.")

    formulas: List[Tuple[float, ...]] = []
    for i, entry in enumerate(data):
        if isinstance(entry, dict) and "formula" in entry:
            try:
                formulas.append(parse_formula(entry["formula"]))
            except Exception as e:
                logging.warning(f"Skip entry {i} due to bad formula: {entry.get('formula')} ({e})")
        else:
            # allow bare list of formulas
            try:
                formulas.append(parse_formula(entry))
            except Exception as e:
                logging.warning(f"Skip entry {i} due to bad entry: {e}")
    # de-dup keep order
    seen = set()
    uniq = []
    for f in formulas:
        if f not in seen:
            uniq.append(f)
            seen.add(f)
    return uniq


def load_layer_masks(layer_masks_path: Path) -> List[Any]:
    data = json.loads(layer_masks_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("layer_masks.json must be a JSON list.")
    return data


def evaluate_one(
    args: argparse.Namespace,
    formula: Tuple[float, ...],
    layer_mask: Any,
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Returns (score, mean_topk, raw_results_minimal)
    """
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
                layer_mask,
            ],
        },
    }

    results = evaluator.simple_evaluate(
        model="local-completions",
        model_args=model_args_dict,
        tasks=["gsm8k_cot"],
        limit=args.num_samples if args.num_samples > 0 else None,
    )

    # score
    try:
        score = float(results["results"]["gsm8k_cot"]["exact_match,flexible-extract"])
    except Exception:
        score = 0.0

    # mean_topk
    try:
        mean_topk = float(results["topk"]["mean_topk"])
    except Exception:
        mean_topk = float("inf")

    # store a tiny bit of raw info helpful for debugging
    raw_min = {
        "task": "gsm8k_cot",
        "metric": "exact_match,flexible-extract",
        "has_topk": bool(results.get("topk")),
    }
    return score, mean_topk, raw_min


def main():
    p = argparse.ArgumentParser(
        description="Evaluate Pareto formulas across all layer_mask choices (cross product).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # inputs
    p.add_argument("--pareto", type=str, required=True, help="Path to pareto_frontier.json")
    p.add_argument("--layer_masks", type=str, required=True, help="Path to layer_masks.json (JSON list)")

    # output
    p.add_argument("--out", type=str, default="pareto_x_layer_masks_eval.json", help="Output JSON path")
    p.add_argument("--resume", action="store_true", help="Resume from existing --out, skip finished items")

    # lm_eval / local-completions
    p.add_argument("--base_url", type=str, default="http://127.0.0.1:8000/v1/completions")
    p.add_argument("--num_concurrent", type=int, default=32)
    p.add_argument("--max_retries", type=int, default=3)
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-30B-A3B")
    p.add_argument("--num_samples", type=int, default=-1, help="lm_eval limit; -1 for all")

    args = p.parse_args()

    pareto_path = Path(args.pareto)
    layer_masks_path = Path(args.layer_masks)
    out_path = Path(args.out)

    formulas = load_pareto_formulas(pareto_path)
    layer_masks = load_layer_masks(layer_masks_path)

    logging.info(f"Loaded {len(formulas)} Pareto formulas from {pareto_path}")
    logging.info(f"Loaded {len(layer_masks)} layer_mask choices from {layer_masks_path}")
    logging.info(f"Total evaluations (cross product): {len(formulas) * len(layer_masks)}")

    # Load existing output if resume
    out_data: Dict[str, Any] = {"meta": {}, "groups": []}
    done_keys = set()

    if args.resume and out_path.exists():
        try:
            out_data = json.loads(out_path.read_text(encoding="utf-8"))
            for g in out_data.get("groups", []):
                for item in g.get("evals", []):
                    k = item.get("key")
                    if k:
                        done_keys.add(k)
            logging.info(f"Resume enabled. Found {len(done_keys)} completed evals in {out_path}")
        except Exception as e:
            logging.warning(f"Failed to load existing out file, will start fresh: {e}")
            out_data = {"meta": {}, "groups": []}
            done_keys = set()

    # Build an index for groups by formula_str
    groups_by_formula: Dict[str, Dict[str, Any]] = {}
    for g in out_data.get("groups", []):
        fstr = g.get("formula")
        if isinstance(fstr, str):
            groups_by_formula[fstr] = g

    # meta
    out_data["meta"] = {
        "pareto": str(pareto_path),
        "layer_masks": str(layer_masks_path),
        "model_name": args.model_name,
        "base_url": args.base_url,
        "num_concurrent": args.num_concurrent,
        "max_retries": args.max_retries,
        "num_samples": args.num_samples,
    }

    # evaluate
    total = len(formulas) * len(layer_masks)
    count = 0
    for formula in formulas:
        formula_str = str(tuple(float(x) for x in formula))
        group = groups_by_formula.get(formula_str)
        if group is None:
            group = {"formula": formula_str, "evals": []}
            out_data["groups"].append(group)
            groups_by_formula[formula_str] = group

        for layer_mask in layer_masks:
            count += 1
            key = _make_key(formula, layer_mask)
            if key in done_keys:
                if count % 20 == 0:
                    logging.info(f"[{count}/{total}] skipped (resume)")
                continue

            logging.info(f"[{count}/{total}] evaluating formula={formula_str} layer_mask={layer_mask}")

            score, mean_topk, raw_min = evaluate_one(args, formula, layer_mask)

            entry = {
                "key": key,
                "layer_mask": layer_mask,
                "score": score,
                "mean_topk": mean_topk,
                "obj_1_minus_score": 1.0 - score,
                "obj_mean_topk": mean_topk,
                "raw_min": raw_min,
            }
            group["evals"].append(entry)
            done_keys.add(key)

            # persist frequently (safe if crash)
            out_path.write_text(json.dumps(out_data, indent=2, ensure_ascii=False), encoding="utf-8")

            # small console summary
            logging.info(f"  -> score={score:.4f} mean_topk={mean_topk}")

    # sort each group by (1-score, mean_topk) to match your minimization view
    for g in out_data["groups"]:
        g["evals"] = sorted(
            g.get("evals", []),
            key=lambda d: (float(d.get("obj_1_minus_score", 1.0)), float(d.get("obj_mean_topk", float("inf")))),
        )

    out_path.write_text(json.dumps(out_data, indent=2, ensure_ascii=False), encoding="utf-8")
    logging.info(f"All done. Results saved to {out_path}")

    # Print top-1 per formula for quick look
    print("\n=== best layer_mask per Pareto formula (by (1-score, mean_topk)) ===")
    for g in out_data["groups"]:
        if not g["evals"]:
            continue
        best = g["evals"][0]
        print(
            f"formula={g['formula']} | best_layer_mask={best['layer_mask']} "
            f"| score={best['score']:.4f} mean_topk={best['mean_topk']}"
        )


if __name__ == "__main__":
    main()
