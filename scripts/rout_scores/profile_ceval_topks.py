import torch
import torch.nn.functional as F
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import pandas as pd
import numpy as np
import os

# --- 1. Configuration ---
MODEL_NAME = "Qwen/Qwen3-30B-A3B"
INPUT_FILE = "data/dyn_topk/dump_topks/ceval_results_with_top_k.jsonl"
OUTPUT_DIR_ROUTING_SCORES = "data/dyn_topk/dump_topks/routing_scores_from_ceval_topk"
MAX_SAMPLES = 256  # Set to -1 to process all samples, or a positive integer for a subset

def main():
    """
    Processes a JSONL file to reconstruct sequences, runs a model forward pass to get router logits,
    and saves the routing scores to Parquet files.
    """
    # --- 2. Initialization ---
    print("Initializing model and tokenizer...")
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR_ROUTING_SCORES, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
        trust_remote_code=True
    )
    # Configure model to output router logits and disable cache for full sequence processing
    model.config.output_router_logits = True
    model.config.use_cache = False
    print("Model and tokenizer initialized.")

    # --- 3. Data Loading ---
    print(f"Loading data from {INPUT_FILE}...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        all_requests = [json.loads(line) for line in f]
    
    if MAX_SAMPLES > 0:
        all_requests = all_requests[:MAX_SAMPLES]
    print(f"Loaded {len(all_requests)} requests.")

    # --- 4. Processing and Prefill ---
    for entry_idx, request_data in tqdm(enumerate(all_requests), total=len(all_requests), desc="Processing Requests"):
        # --- a. Reconstruct sequences ---
        prompt_ids = request_data.get('prompt_ids', [])
        think_ids = request_data.get('think_part_ids', [])
        answer_ids = request_data.get('answer_part_ids', [])
        
        # The full token sequence for the model input
        full_sequence_ids = prompt_ids + think_ids + answer_ids
        if not full_sequence_ids:
            print(f"Warning: Skipping entry {entry_idx} due to empty token sequence.")
            continue

        # The top_k values correspond only to the generated parts (think and answer).
        think_top_ks = request_data.get('think_part_top_ks', [])
        answer_top_ks = request_data.get('answer_part_top_ks', [])
        
        # The model expects top_k values for each token. For prompt tokens, which are not generated,
        # we'll use the default number of experts per token as their top_k value.
        num_experts_per_tok = model.config.num_experts_per_tok
        
        prompt_len = len(prompt_ids)
        
        # Create the full top_k list.
        # Prompt tokens get the default value.
        # Generated tokens (think + answer) get their recorded top_k values.
        token_top_ks = [num_experts_per_tok] * prompt_len + think_top_ks + answer_top_ks
        
        # Ensure lengths match
        if len(token_top_ks) != len(full_sequence_ids):
            print(f"Warning: Mismatch in sequence length ({len(full_sequence_ids)}) and top_k length ({len(token_top_ks)}) for entry {entry_idx}. Skipping.")
            continue

        input_ids_tensor = torch.tensor([full_sequence_ids], device=model.device)
        token_top_ks_tensor = torch.tensor([token_top_ks], device=model.device)

        # --- b. Model Prefill for Router Logits ---
        try:
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids_tensor,
                    token_top_ks=token_top_ks_tensor  # Pass the custom top_k values
                )
        except Exception as e:
            print(f"Error during model forward pass for entry {entry_idx}: {e}")
            continue

        # --- c. Process and Dump Router Logits ---
        if hasattr(outputs, 'router_logits') and outputs.router_logits is not None:
            # Stack router logits from all MoE layers: (num_layers, batch_size, num_tokens, num_experts)
            # Squeeze batch_size dim since it's 1
            router_logits_tensor = torch.stack(outputs.router_logits, dim=0).squeeze(1) # Shape: (num_layers, num_tokens, num_experts)
            
            # Apply softmax to get routing scores
            routing_scores_tensor = F.softmax(router_logits_tensor, dim=-1)

            # Apply top-k mask
            # routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
            # if token_top_ks is not None:
            #     # Mask out the expert weights that are not selected
            #     drop_expert_mask = torch.arange(0, self.top_k, device=routing_weights.device) >= token_top_ks[:, None]
            #     routing_weights.masked_fill_(drop_expert_mask, 0.0)
            # if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
            #     routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
            
            routing_weights, selected_experts = torch.topk(routing_scores_tensor, k=num_experts_per_tok, dim=-1)
            # Mask out experts beyond token_top_ks
            for tok_idx in range(routing_weights.size(1)):
                top_k_val = token_top_ks[tok_idx]
                if top_k_val < num_experts_per_tok:
                    routing_weights[:, tok_idx, top_k_val:] = 0.0
            # Norm after masking
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
            routing_scores_tensor = routing_weights

            if routing_scores_tensor.numel() == 0:
                print(f"Warning: router_logits tensor for entry {entry_idx} is empty. Skipping Parquet generation.")
                continue

            # Convert to NumPy array for efficient DataFrame creation
            routing_scores_np = routing_scores_tensor.to(device='cpu', dtype=torch.float32).numpy()
            del router_logits_tensor, routing_scores_tensor

            # Create index arrays for each dimension
            layer_indices, token_indices, _ = np.indices(routing_scores_np.shape)

            # Flatten all arrays for DataFrame columns
            flat_layer_ids = layer_indices.ravel()
            flat_token_positions = token_indices.ravel()
            flat_expert_ids = selected_experts.flatten().cpu().numpy()
            flat_routing_scores = routing_scores_np.ravel()
            
            # Create request_id array, repeating for all routing entries
            total_routing_entries = routing_scores_np.size
            request_id_array = np.full(total_routing_entries, entry_idx, dtype=np.int32)

            # Create DataFrame
            df_routing = pd.DataFrame({
                'request_id': request_id_array,
                'token_position_in_sequence': flat_token_positions,
                'layer_id': flat_layer_ids,
                'expert_id': flat_expert_ids,
                'routing_score': flat_routing_scores
            })
            
            # Save to Parquet
            parquet_file_path = os.path.join(OUTPUT_DIR_ROUTING_SCORES, f"request_{entry_idx}_routing_scores.parquet")
            df_routing.to_parquet(parquet_file_path, index=False)
        else:
            print(f"Warning: router_logits not found in outputs for entry {entry_idx}.")

    print(f"\nProcessing complete. Routing scores saved to {OUTPUT_DIR_ROUTING_SCORES}")

if __name__ == "__main__":
    main()
