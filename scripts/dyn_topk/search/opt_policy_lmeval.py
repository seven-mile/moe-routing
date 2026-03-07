import argparse
import json
import logging
import numpy as np
import os
import torch
from pathlib import Path
from itertools import accumulate
from typing import List, Tuple, Dict, Any, Sequence

# --- Skopt Imports ---
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from skopt.callbacks import CheckpointSaver

# --- New Imports for lm_eval ---
from lm_eval import evaluator
# from lm_eval.utils import print_tokens_per_second # This is often used for logging but not strictly necessary for the core logic

# --- Basic Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
# Suppress noisy logging from sub-libraries
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)

# --- Core Utility Functions (Moved or removed) ---
# Removed PPL-related utilities: calculate_token_losses, get_assisted_topks.
# Since lm_eval handles the model/tokenizer loading, the dynamic config part is also simplified.

def convert_point_to_formula(point: Sequence[float]) -> Tuple[float, ...]:
    """Converts a skopt point to a monotonically non-increasing tuple."""
    return tuple(accumulate(point, lambda x, y: x * y))

def convert_formula_to_point(formula: Sequence[float]) -> Tuple[float, ...]:
    """Converts a formula back to the skopt point representation (p0, p1, p2, p3)."""
    p0, p1, p2, p3 = formula
    return p0, p1 / p0, p2 / p1, p3 / p2

# --- New Evaluation Logic using lm_eval ---

def evaluate_with_lmeval(args: argparse.Namespace, formula: Tuple[float, ...]) -> float:
    """
    Calls lm_eval's evaluation core with the given speculative decoding formula.
    Returns the gsm8k_cot accuracy score.
    """
    logging.info(f"Setting up lm_eval for formula: {formula}")

    # 1. Construct the MODEL_ARGS dictionary
    # The 'base_url' and 'model' must match the execution environment
    # 'model' is just a placeholder name for the local-completions model
    model_args_dict = {
        "model": args.model_name, # e.g., "Qwen/Qwen3-30B-A3B"
        "base_url": args.base_url, # e.g., "http://0.0.0.0:7000/v1/completions"
        "num_concurrent": args.num_concurrent,
        "max_retries": args.max_retries,
        "tokenized_requests": False,
        "assisted_action": {
            "file": "configs/ppl_to_ks.py",
            "function": "spec_with_list_layer_range",
            "args": [
                list(formula), # Strategy values go here
                [20, 25]       # Layer range goes here (fixed as per original command)
            ]
        },
    }
    
    # 2. Perform the evaluation
    # lm_eval.evaluator.simple_evaluate is the core function
    # The 'local-completions' model needs 'model_args' passed correctly, which is done by the string above.
    
    results = evaluator.simple_evaluate(
        model="local-completions",
        model_args=model_args_dict,
        tasks=["gsm8k_cot"],
        # Limit evaluation to a smaller subset if needed (optional optimization)
        limit=args.num_samples if args.num_samples > 0 else None,
        # Other necessary lm_eval parameters might be required depending on the environment
    )

    # 3. Extract the score
    try:
        score = results["results"]["gsm8k_cot"]["exact_match,flexible-extract"]
        logging.info(f"lm_eval finished. gsm8k_cot Accuracy: {score:.4f}")
        return score
    except KeyError:
        logging.error("Could not find 'gsm8k_cot' accuracy in lm_eval results.")
        return 0.0 # Return 0.0 or a very low score on failure

# --- Main Optimization Logic (Modified) ---

def run_optimization(args: argparse.Namespace):
    """Sets up and runs the Bayesian optimization loop using lm_eval score."""
    
    # Removed all model and tokenizer loading (model, spec_model, tokenizer, dataset)
    # as lm_eval handles this internally for the specified model/base_url.
    
    # --- Skopt Objective Function ---
    
    # 1. Define the continuous search space
    dimensions = [
        Real(1.0, 10.0, name='p0'),
        Real(0.0, 1.0, name='p1'),
        Real(0.0, 1.0, name='p2'),
        Real(0.0, 1.0, name='p3'),
    ]

    # Load previous results
    results = []
    output_path = Path(args.output_file)
    if output_path.exists():
        try:
            with open(output_path, 'r') as f:
                results = json.load(f)
            logging.info(f"Loaded {len(results)} existing results from {output_path}")
        except json.JSONDecodeError:
            logging.warning(f"Could not parse {output_path}. Starting fresh.")

    @use_named_args(dimensions)
    def objective(**params):
        """
        The objective function that skopt will minimize.
        It calls lm_eval, gets the score, and returns (1 - score).
        """
        point = tuple(params.values())
        formula = convert_point_to_formula(point)
        
        # Check monotonicity (The formula's thresholds must be non-increasing)
        if not all(formula[i] >= formula[i+1] for i in range(len(formula) - 1)):
            logging.warning(f"Skipping invalid (non-monotonic) formula: {formula}")
            # Return a high value to penalize this invalid point (since we minimize)
            return 1.0
        
        formula_str = str(formula)
        # Check if already evaluated
        for res in results:
            if res['formula'] == formula_str:
                logging.info(f"Formula {formula} already evaluated. Score: {res['score']:.4f}. Skipping.")
                # Return the corresponding minimization target (1 - score)
                return 1.0 - res['score']
        
        logging.info(f"Now evaluating formula: {formula}")

        # --- NEW EVALUATION STEP ---
        score = evaluate_with_lmeval(args, formula)
        
        # --- Store Result ---
        result_entry = {"formula": formula_str, "score": score}
        results.append(result_entry)
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
            
        # The objective function must be MINIMIZED by skopt.
        # We want to MAXIMIZE the score, so we return (1 - score).
        minimization_target = 1.0 - score
        logging.info(f"--> Result | Formula: {formula}, Score: {score:.4f}, Target to Minimize: {minimization_target:.4f}")
        
        return minimization_target

    # --- Main Optimization Flow ---
    
    # 1. Calculate Baseline Score (optional)
    logging.info("Checking baseline score (formula='full')...")
    baseline_score = None
    for res in results:
        if res.get("formula") == "baseline":
            baseline_score = res["score"]
            break

    if baseline_score is None:
        # Evaluate baseline (full speculative decoding is equivalent to a very high/max threshold)
        # We can use the formula that corresponds to all experts being active, e.g., (100.0, 100.0, 100.0, 100.0)
        # However, for 'local-completions', we might just skip the 'dyn_assisted_action_config'
        # or use a very permissive formula. For simplicity and aligning with the 'baseline' concept,
        # we'll use a standard, non-optimized strategy if needed, but since we're optimizing,
        # a true "baseline" might be "full model" which isn't easy to represent here.
        # Let's use the provided first example as an initial point instead of a formal "baseline"
        # or rely on the optimization process to find good points.
        pass # Skipping formal baseline calculation as it's complex to define in the new context
        
    # 2. Setup checkpointing to save/resume optimization
    checkpoint_path = "skopt_checkpoint.pkl"
    checkpoint_saver = CheckpointSaver(checkpoint_path, store_objective=False)
    
    # 3. Run Bayesian Optimization
    logging.info("Starting Bayesian Optimization...")
    
    # Prepare warm-start data from existing results
    x0, y0 = [], []
    for res in results:
        if res['formula'] not in ('baseline', 'full'):
            try:
                formula = eval(res['formula'])
                point = convert_formula_to_point(formula)
                x0.append(point)
                y0.append(1.0 - res['score']) # Minimization target
            except:
                continue
    
    # Add the example from the prompt as an initial point (if not already evaluated)
    example_formula = (10.0, 6.58, 1.275, 1.0)
    example_point = convert_formula_to_point(example_formula)
    
    # Only add the example point if it's not in the loaded x0
    if example_point not in x0:
         example_formula_str = str(example_formula)
         already_evaluated = any(res['formula'] == example_formula_str for res in results)
         if not already_evaluated:
             # Add the formula to the initial points for skopt to evaluate first
             x0.append(example_point)
             # y0 is kept None/empty for this point so skopt evaluates it
             
    opt_result = gp_minimize(
        func=objective,
        dimensions=dimensions,
        n_calls=args.n_calls,
        n_initial_points=args.n_initial_points,
        random_state=123,
        callback=[checkpoint_saver],
        x0=x0 if x0 else None,
        y0=y0 if y0 else None
    )

    logging.info("Optimization complete.")
    best_minimization_target = opt_result.fun
    best_score = 1.0 - best_minimization_target
    logging.info(f"Best Score found: {best_score:.4f}")
    best_formula = convert_point_to_formula(opt_result.x)
    logging.info(f"Best formula: {best_formula}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optimize speculative decoding strategies with Bayesian Optimization using lm_eval.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # --- Arguments for lm_eval and local-completions Model ---
    # These are needed to construct the MODEL_ARGS string for lm_eval
    parser.add_argument("--base_url", type=str, default="http://0.0.0.0:7000/v1/completions", help="Base URL for the local-completions API.")
    parser.add_argument("--num_concurrent", type=int, default=128, help="Number of concurrent requests for local-completions.")
    parser.add_argument("--max_retries", type=int, default=3, help="Max retries for local-completions requests.")

    # --- Skopt and Optimization Arguments (Kept or modified) ---
    parser.add_argument("--data_file", type=str, default="N/A_lmeval", help="Path to the JSON dataset file (No longer strictly used).")
    parser.add_argument("--output_file", type=str, default="optimization_results_lmeval.json", help="Path to save the results JSON file.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-30B-A3B", help="Main model name (placeholder for local-completions).")
    parser.add_argument("--num_samples", type=int, default=-1, help="Number of samples to use. -1 for all (Passed as 'limit' to lm_eval).")

    # --- New Arguments for Skopt ---
    parser.add_argument("--n_calls", type=int, default=100, help="Total number of evaluations to perform.")
    parser.add_argument("--n_initial_points", type=int, default=10, help="Number of random points to sample before fitting the model.")

    args = parser.parse_args()
        
    run_optimization(args)
