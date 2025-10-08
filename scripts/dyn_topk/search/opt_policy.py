"""
This script uses Bayesian Optimization to find optimal speculative decoding strategies.

It defines a continuous search space for perplexity thresholds and uses the skopt
library to efficiently find parameter combinations that minimize the perplexity (PPL)
of a 'think_part' in a given dataset.

The evaluation is performed in batches. Optimization progress is saved automatically,
allowing the process to be resumed if interrupted.
"""

import argparse
import json
import logging
import numpy as np
import os
import torch
from pathlib import Path
from tqdm import tqdm
from datasets import Dataset
from itertools import accumulate
from torch.nn.functional import cross_entropy
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple, Dict, Any, Sequence

# --- Skopt Imports ---
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from skopt.callbacks import CheckpointSaver

# --- Basic Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- Core Utility Functions (Unchanged from original) ---

def calculate_token_losses(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Calculates the cross-entropy loss for each token (log-perplexity)."""
    # Ensure logits are in float32 for numerical precision.
    logits = logits.float()
    assert logits.shape[:-1] == ids.shape
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = ids[..., 1:].contiguous()
    loss = cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction='none'
    )
    return loss.view(shift_labels.shape)

def get_assisted_topks(
    cfg: Tuple[float, ...], log_ppls: torch.FloatTensor, k: int
) -> torch.LongTensor:
    """Determines the top-k value for each token based on its PPL."""
    ppls = torch.exp(log_ppls)
    ks = torch.full_like(ppls, k, dtype=torch.int64)
    # The cfg thresholds are PPL values that map to a reduced number of experts.
    for threshold, top_k_val in zip(cfg, range(k - 1, -1, -1)):
        ks[ppls < threshold] = top_k_val
    return ks

# --- Main Evaluation Logic ---

def run_optimization(args: argparse.Namespace):
    """Sets up models and runs the Bayesian optimization loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    logging.info("Loading models and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map='auto',
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_cache=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    spec_model = AutoModelForCausalLM.from_pretrained(
        args.spec_model_name,
        device_map='auto',
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_cache=False,
    )
    
    model.eval()
    spec_model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'

    # --- Get Model Configuration Dynamically ---
    base_top_k = getattr(model.config, 'num_experts_per_tok', 1)
    num_layers = model.config.num_hidden_layers
    logging.info(f"Detected model configuration: num_layers={num_layers}, base_top_k={base_top_k}")

    logging.info(f"Loading dataset from {args.data_file}...")
    with open(args.data_file, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    
    if args.num_samples > 0:
        samples = samples[:args.num_samples]

    dataset = Dataset.from_list(samples)

    # Pre-tokenize markers for robust index searching.
    start_marker_ids = tokenizer("<think>", add_special_tokens=False).input_ids
    end_marker_ids = tokenizer("</think>", add_special_tokens=False).input_ids

    def process_batch(
        batch: List[Dict[str, Any]], formula: Tuple | None
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Processes a batch of text samples, runs inference, and returns results.
        (This function is identical to the original script)
        """
        texts = [
            f"{s['question'].strip()}\nA. {s['A']}\nB. {s['B']}\nC. {s['C']}\nD. {s['D']}\n"
            f"<think>{s['think_part']}</think>{s['answer_part']}"
            for s in batch
        ]
        inputs = tokenizer(
            texts, padding=True, truncation=True,
            max_length=args.max_length, return_tensors="pt"
        ).to(device)
        input_ids = inputs.input_ids

        think_mask = torch.zeros(input_ids.shape[0], input_ids.shape[1] - 1, dtype=torch.bool, device=device)
        for i, ids in enumerate(input_ids.tolist()):
            try:
                start_pos = -1
                for j in range(len(ids) - len(start_marker_ids) + 1):
                    if ids[j : j + len(start_marker_ids)] == start_marker_ids:
                        start_pos = j
                        break
                
                end_pos = -1
                for j in range(start_pos, len(ids) - len(end_marker_ids) + 1):
                    if ids[j : j + len(end_marker_ids)] == end_marker_ids:
                        end_pos = j
                        break

                if start_pos != -1 and end_pos != -1:
                    mask_start = start_pos + len(start_marker_ids) - 1
                    mask_end = end_pos - 1
                    think_mask[i, mask_start:mask_end] = True
            except (ValueError, TypeError):
                logging.warning(f"Could not find markers in sample {i}. Skipping its mask.")
                continue
            
        with torch.no_grad():
            if formula is None:
                outputs = model(input_ids=input_ids, attention_mask=inputs.attention_mask)
                losses = calculate_token_losses(outputs.logits, input_ids)
                return losses, think_mask, 0.0

            spec_outputs = spec_model(input_ids=input_ids, attention_mask=inputs.attention_mask)
            spec_losses = calculate_token_losses(spec_outputs.logits, input_ids)

            layer_ks_per_token = torch.full_like(spec_losses, base_top_k, dtype=torch.int64)
            think_part_ks = get_assisted_topks(formula, spec_losses[think_mask], base_top_k)
            
            benefit = 0.0
            if think_part_ks.numel() > 0:
                benefit = (base_top_k - think_part_ks.float().mean()).item() / base_top_k
            
            layer_ks_per_token[think_mask] = think_part_ks
            
            dummy_k = torch.full((layer_ks_per_token.shape[0], 1), base_top_k, dtype=torch.int64, device=device)
            layer_ks = torch.cat([layer_ks_per_token, dummy_k], dim=1)
            
            token_top_ks = torch.stack([layer_ks] * num_layers, dim=0)
            
            reduced_outputs = model(
                input_ids=input_ids,
                attention_mask=inputs.attention_mask,
                token_top_ks=token_top_ks
            )
            reduced_losses = calculate_token_losses(reduced_outputs.logits, input_ids)
            return reduced_losses, think_mask, benefit

    # --- Skopt Objective Function ---
    
    # 1. Define the continuous search space
    dimensions = [
        Real(1.0, 10.0, name='p0'),
        Real(0.0, 1.0, name='p1'),
        Real(0.0, 1.0, name='p2'),
        Real(0.0, 1.0, name='p3'),
    ]

    def convert_point_to_formula(point: Sequence[float]) -> Tuple[float, ...]:
        """Converts a skopt point to a monotonically non-increasing tuple."""
        return tuple(accumulate(point, lambda x, y: x * y))
    
    def convert_formula_to_point(formula: Sequence[float]) -> Tuple[float, ...]:
        """Converts a formula back to the skopt point representation."""
        p1, p2, p3, p4 = formula
        return p1, p2 / p1, p3 / p2, p4 / p3

    # Load previous results to avoid re-evaluation and inform the optimizer
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
        It takes a formula, evaluates its PPL, and returns it.
        """
        point = tuple(params.values())
        formula = convert_point_to_formula(point)
        
        if not all(formula[i] >= formula[i+1] for i in range(len(formula) - 1)):
            logging.warning(f"Skipping invalid (non-monotonic) formula: {formula}")
            return 1e10 
        
        formula_str = str(formula)
        if any(res['formula'] == formula_str for res in results):
            logging.info(f"Formula {formula} already evaluated. Skipping.")
            for res in results:
                if res['formula'] == formula_str:
                    return res['avg_ppl']
            
        logging.info(f"Now evaluating formula: {formula}")

        formula_losses, benefits = [], []
        pbar_desc = f"Eval {formula[0]:.2f}, {formula[1]:.2f}, ..."
        for i in tqdm(range(0, len(dataset), args.batch_size), desc=pbar_desc, leave=False):
            batch = [dataset[j] for j in range(i, min(i + args.batch_size, len(dataset)))]
            losses, think_mask, benefit = process_batch(batch, formula)
            benefits.append(benefit)
            masked_losses = losses[think_mask]
            if masked_losses.numel() > 0:
                formula_losses.append(masked_losses)

        if not formula_losses:
            avg_ppl = float('inf')
        else:
            mean_loss = torch.cat(formula_losses).mean()
            avg_ppl = torch.exp(mean_loss).item()
        
        avg_benefit = sum(benefits) / len(benefits) if benefits else 0.0
        
        result_entry = {"formula": formula_str, "avg_ppl": avg_ppl, "benefit": avg_benefit}
        results.append(result_entry)
        
        logging.info(f"--> Result | Formula: {formula}, Avg PPL: {avg_ppl:.4f}, Benefit: {avg_benefit:.6f}")
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        return avg_ppl

    # --- Main Optimization Flow ---
    
    # 1. Calculate Baseline PPL (optional but good for context)
    logging.info("Calculating baseline perplexity (if not already present)...")
    baseline_ppl = None
    for res in results:
        if res.get("formula") == "baseline":
            baseline_ppl = res["avg_ppl"]
            break

    if baseline_ppl is None:
        baseline_losses = []
        for i in tqdm(range(0, len(dataset), args.batch_size), desc="Baseline Calc"):
            batch = [dataset[j] for j in range(i, min(i + args.batch_size, len(dataset)))]
            losses, think_mask, _ = process_batch(batch, None)
            masked_losses = losses[think_mask]
            if masked_losses.numel() > 0:
                baseline_losses.append(masked_losses)

        if baseline_losses:
            mean_loss = torch.cat(baseline_losses).mean()
            baseline_ppl = torch.exp(mean_loss).item()
            results.append({"formula": "baseline", "avg_ppl": baseline_ppl, "benefit": 0.0})
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=4)
        else:
            baseline_ppl = float('inf')
    logging.info(f"Baseline (Full) Perplexity: {baseline_ppl:.4f}")

    # 2. Setup checkpointing to save/resume optimization
    checkpoint_path = "skopt_checkpoint.pkl"
    checkpoint_saver = CheckpointSaver(checkpoint_path, store_objective=False)
    
    # 3. Run Bayesian Optimization
    logging.info("Starting Bayesian Optimization...")
    
    # Prepare warm-start data from existing results
    x0, y0 = [], []
    for res in results:
        if res['formula'] != 'baseline':
            try:
                formula = eval(res['formula'])
                point = convert_formula_to_point(formula)
                x0.append(point)
                y0.append(res['avg_ppl'])
            except:
                continue
    
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
    logging.info(f"Best PPL found: {opt_result.fun:.4f}")
    best_formula = convert_point_to_formula(opt_result.x)
    logging.info(f"Best formula: {best_formula}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optimize speculative decoding strategies with Bayesian Optimization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # --- Original Arguments ---
    parser.add_argument("--data_file", type=str, required=True, help="Path to the JSON dataset file.")
    parser.add_argument("--output_file", type=str, default="optimization_results.json", help="Path to save the results JSON file.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-30B-A3B", help="Main model name or path.")
    parser.add_argument("--spec_model_name", type=str, default="Qwen/Qwen3-0.6B", help="Speculative model name or path.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation.")
    parser.add_argument("--max_length", type=int, default=2048, help="Max sequence length for tokenizer.")
    parser.add_argument("--num_samples", type=int, default=32, help="Number of samples to use. -1 for all.")
    parser.add_argument("--devices", type=str, default="0", help="Comma-separated list of GPU device IDs to use.")

    # --- New Arguments for Skopt ---
    parser.add_argument("--n_calls", type=int, default=100, help="Total number of evaluations to perform.")
    parser.add_argument("--n_initial_points", type=int, default=10, help="Number of random points to sample before fitting the model.")

    args = parser.parse_args()
    
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
        
    run_optimization(args)
