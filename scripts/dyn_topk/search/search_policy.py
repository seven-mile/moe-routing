"""
This script evaluates the performance of various speculative decoding strategies.

It works by taking a base strategy (a tuple of perplexity thresholds) and
generating a search space of similar strategies. Each strategy is then evaluated
against a dataset by calculating the perplexity (PPL) of a specific 'think_part'
in each sample.

The evaluation is performed in batches, with tokenization and processing handled
dynamically within the main evaluation loop to ensure correctness. Results are
saved incrementally to a JSON file.
"""

import argparse
import json
import logging
import numpy as np
import os
import torch
import itertools
from pathlib import Path
from tqdm import tqdm
from datasets import Dataset
from torch.nn.functional import cross_entropy
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Sequence, List, Tuple, Dict, Any

# --- Basic Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- Core Utility Functions ---

def calculate_token_losses(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Calculates the cross-entropy loss for each token (log-perplexity)."""
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
    cfg: Sequence[float], log_ppls: torch.FloatTensor, k: int
) -> torch.LongTensor:
    """Determines the top-k value for each token based on its PPL."""
    ppls = torch.exp(log_ppls)
    ks = torch.full_like(ppls, k, dtype=torch.int64)
    # The cfg thresholds are PPL values that map to a reduced number of experts.
    for threshold, top_k_val in zip(cfg, range(k - 1, -1, -1)):
        ks[ppls < threshold] = top_k_val
    return ks

def generate_formulas(base: Tuple, space: dict) -> List[Tuple]:
    """Generates a list of unique, monotonically decreasing formulas."""
    if not base:
        return []
    
    options = [space.get(i, [v]) for i, v in enumerate(base)]
    candidates = list(itertools.product(*options))
    
    # A formula is valid only if its threshold values are non-increasing.
    valid_formulas = {
        f for f in candidates if all(f[i] >= f[i+1] for i in range(len(f) - 1))
    }
    
    sorted_formulas = sorted(list(valid_formulas))
    if base in sorted_formulas:
        sorted_formulas.remove(base)
        sorted_formulas.insert(0, base)
        
    logging.info(f"Generated {len(sorted_formulas)} valid formulas to test.")
    return sorted_formulas

# A helper function to generate float ranges more cleanly.
def frange(start: float, stop: float, step: float, decimals: int = 3) -> List[float]:
    """Generates a range of floats with a specified number of decimal places."""
    return np.round(np.arange(start, stop, step), decimals).tolist()

# --- Main Evaluation Logic ---

def run_evaluation(args: argparse.Namespace):
    """Sets up models and runs the main evaluation loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    logging.info("Loading models and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map='auto',
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    spec_model = AutoModelForCausalLM.from_pretrained(
        args.spec_model_name,
        device_map='auto',
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
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
        Tokenization and masking are handled here to ensure correctness for each batch.
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

        # Find think_part indices by searching for marker tokens. This is more
        # robust than calculating lengths of substrings, which can be misleading
        # due to context-dependent tokenization.
        think_mask = torch.zeros(input_ids.shape[0], input_ids.shape[1] - 1, dtype=torch.bool, device=device)
        for i, ids in enumerate(input_ids.tolist()):
            try:
                # Find the position of the start marker token sequence
                start_pos = -1
                for j in range(len(ids) - len(start_marker_ids) + 1):
                    if ids[j : j + len(start_marker_ids)] == start_marker_ids:
                        start_pos = j
                        break
                
                # Find the position of the end marker token sequence
                end_pos = -1
                for j in range(start_pos, len(ids) - len(end_marker_ids) + 1):
                     if ids[j : j + len(end_marker_ids)] == end_marker_ids:
                        end_pos = j
                        break

                if start_pos != -1 and end_pos != -1:
                    # The mask applies to the loss tensor, which is one token shorter.
                    # The loss for the token *after* the start marker is what we want first.
                    mask_start = start_pos + len(start_marker_ids) - 1
                    mask_end = end_pos - 1
                    think_mask[i, mask_start:mask_end] = True
            except (ValueError, TypeError):
                logging.warning(f"Could not find markers in sample {i}. Skipping its mask.")
                continue
            
        with torch.no_grad():
            if formula is None:  # Baseline calculation
                outputs = model(input_ids=input_ids, attention_mask=inputs.attention_mask)
                losses = calculate_token_losses(outputs.logits, input_ids)
                return losses, think_mask, 0.0

            spec_outputs = spec_model(input_ids=input_ids, attention_mask=inputs.attention_mask)
            spec_losses = calculate_token_losses(spec_outputs.logits, input_ids)

            layer_ks_per_token = get_assisted_topks(formula, spec_losses, base_top_k)
            dummy_k = torch.full((layer_ks_per_token.shape[0], 1), base_top_k, dtype=torch.int64, device=device)
            layer_ks = torch.cat([layer_ks_per_token, dummy_k], dim=1)
            
            # Assume all layers use the same k-selection strategy.
            token_top_ks = torch.stack([layer_ks] * num_layers, dim=0)

            benefit = (token_top_ks.sum() / (token_top_ks.numel() * base_top_k)).item()
            
            reduced_outputs = model(
                input_ids=input_ids,
                attention_mask=inputs.attention_mask,
                token_top_ks=token_top_ks
            )
            reduced_losses = calculate_token_losses(reduced_outputs.logits, input_ids)
            return reduced_losses, think_mask, benefit

    # --- Main Evaluation Flow ---
    
    # 1. Generate formulas to test
    base_formula = (6, 1.17, 1.07, 1.07)
    search_space = {
        # Dimension 0 (Base: 6.0): Explores integer values around the base.
        0: [4.0, 5.0, 6.0, 7.0, 8.0],
        # Dimension 1 (Base: 1.17): Denser search with a 0.01 step.
        1: frange(1.12, 1.22, 0.01),
        # Dimension 2 (Base: 1.07): Denser search with a 0.01 step.
        2: frange(1.04, 1.14, 0.01),
        # Dimension 3 (Base: 1.07): Varies the final threshold.
        3: frange(1.02, 1.10, 0.02),
    }
    formulas_to_test = generate_formulas(base_formula, search_space)
    # You can add this print statement after the definition to verify the size.
    print(
        f"Search space configured. Total candidate points: "
        f"{len(formulas_to_test)}"
    )
    
    # 2. Load previous results if they exist
    results = []
    output_path = Path(args.output_file)
    if output_path.exists():
        try:
            with open(output_path, 'r') as f:
                results = json.load(f)
            logging.info(f"Loaded {len(results)} existing results from {output_path}")
        except json.JSONDecodeError:
            logging.warning(f"Could not parse {output_path}. Starting with fresh results.")

    # 3. Calculate Baseline PPL
    logging.info("Calculating baseline perplexity...")
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
    else:
        baseline_ppl = float('inf')
    logging.info(f"Baseline (Full) Perplexity: {baseline_ppl:.4f}")

    # 4. Evaluate each formula
    for formula in tqdm(formulas_to_test, desc="Testing Formulas"):
        formula_str = str(formula)
        if any(res['formula'] == formula_str for res in results):
            continue

        formula_losses, benefits = [], []
        for i in tqdm(range(0, len(dataset), args.batch_size), desc=f"Formula {formula}", leave=False):
            batch = [dataset[j] for j in range(i, min(i + args.batch_size, len(dataset)))]
            losses, think_mask, benefit = process_batch(batch, formula)
            benefits.append(benefit)
            masked_losses = losses[think_mask]
            if masked_losses.numel() > 0:
                 formula_losses.append(masked_losses)

        if formula_losses:
            mean_loss = torch.cat(formula_losses).mean()
            avg_ppl = torch.exp(mean_loss).item()
        else:
            avg_ppl = float('inf')
        
        avg_benefit = sum(benefits) / len(benefits) if benefits else 0.0
        
        results.append({"formula": formula_str, "avg_ppl": avg_ppl, "benefit": avg_benefit})
        logging.info(f"Formula: {formula}, Avg PPL: {avg_ppl:.4f}, Benefit: {avg_benefit:.6f}")
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
            
    logging.info("Evaluation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate speculative decoding strategies.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data_file", type=str, required=True, help="Path to the JSON dataset file.")
    parser.add_argument("--output_file", type=str, default="evaluation_results.json", help="Path to save the results JSON file.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-30B-A3B", help="Main model name or path.")
    parser.add_argument("--spec_model_name", type=str, default="Qwen/Qwen3-0.6B", help="Speculative model name or path.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation.")
    parser.add_argument("--max_length", type=int, default=2048, help="Max sequence length for tokenizer.")
    parser.add_argument("--num_samples", type=int, default=32, help="Number of samples to use. -1 for all.")
    parser.add_argument("--devices", type=str, default="0", help="Comma-separated list of GPU device IDs to use.")
    
    args = parser.parse_args()
    
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
        
    run_evaluation(args)
