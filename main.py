from transformers import AutoModelForCausalLM, AutoTokenizer

from datasets import load_dataset

from tqdm import tqdm

import torch
import torch.nn.functional as F

import pdb
import numpy as np
import pandas as pd
import os

def main():
    TOTAL_ENTRIES = 256

    main_device = 'cuda:0'

    # Create directories for output
    os.makedirs("data/request_text", exist_ok=True)
    os.makedirs("data/routing_scores", exist_ok=True)

    dataset = load_dataset("lmsys/lmsys-chat-1m", split="train")
    active_data = dataset.take(TOTAL_ENTRIES)

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-30B-A3B", device_map=main_device, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.config.output_router_logits = True
    model.config.use_cache = False # Important for getting all router logits

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B", trust_remote_code=True)

    # Data storage for Table 1
    all_request_data_table1 = []

    for entry_idx, entry in tqdm(enumerate(active_data), total=TOTAL_ENTRIES, desc="Processing Entries"):
        conv = entry['conversation']
        chat_string = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)

        inputs = tokenizer(chat_string, padding=True, padding_side='left', return_tensors="pt").to(model.device)

        input_ids_tensor = inputs.input_ids
        input_length = input_ids_tensor.size(1)

        # Store data for Table 1
        for token_pos in range(input_length):
            all_request_data_table1.append({
                'request_id': entry_idx,
                'token_position_in_sequence': token_pos,
                'token_actual_id': input_ids_tensor[0, token_pos].item()
            })

        # Get outputs including router_logits
        # Ensure the model is configured to output router_logits
        # The model call might need to be adjusted if generate is used,
        # but here we are doing a forward pass.
        with torch.no_grad():
            outputs = model(**inputs)
        
        # router_logits shape: (batch_size, num_tokens, num_layers, num_experts)
        # Since we process one entry at a time, batch_size is 1.
        # For Qwen2 MoE, num_layers might be the number of MoE layers, not total layers.
        # And num_experts is specific to the MoE architecture.
        
        if hasattr(outputs, 'router_logits') and outputs.router_logits is not None:
            # Squeeze batch dimension if it's 1, assuming batch_size is 1 for this loop
            router_logits_tensor = torch.stack(outputs.router_logits, dim=0) # Shape: (num_layers, num_tokens, num_experts)
            
            # Softmax across experts to get routing scores
            router_logits_tensor = F.softmax(router_logits_tensor, dim=-1)
            
            # Check if the router_logits tensor is empty (e.g., if num_tokens from model output is 0)
            if router_logits_tensor.numel() == 0:
                print(f"Warning: router_logits tensor for entry {entry_idx} is empty. Skipping Parquet generation for this entry.")
            else:
                num_layers, num_tokens, num_experts = router_logits_tensor.shape
                
                # Efficiently create DataFrame for Table 2 using NumPy vectorization.
                # 'token_position_in_sequence' in the DataFrame corresponds to the token index
                # within the model's input sequence (which router_logits are aligned with).
                
                # Convert router_logits tensor to NumPy array on CPU
                router_logits_np = router_logits_tensor.to(device='cpu', dtype=torch.float32).numpy()
                del router_logits_tensor

                # Create index arrays for each dimension
                layer_indices_np, token_indices_np, expert_indices_np = np.indices(router_logits_np.shape)

                # Flatten all arrays to 1D for DataFrame columns
                flat_layer_ids = layer_indices_np.ravel()
                flat_token_positions = token_indices_np.ravel()
                flat_expert_ids = expert_indices_np.ravel()
                flat_routing_scores = router_logits_np.ravel()
                
                # Create request_id array, repeating entry_idx for all routing entries
                total_routing_entries = router_logits_np.size
                request_id_array = np.full(total_routing_entries, entry_idx, dtype=np.int32) # Assuming entry_idx fits int32

                # Create DataFrame
                df_table2 = pd.DataFrame({
                    'request_id': request_id_array,
                    'token_position_in_sequence': flat_token_positions,
                    'layer_id': flat_layer_ids,
                    'expert_id': flat_expert_ids,
                    'routing_score': flat_routing_scores
                })
                
                # Optional: Convert routing_score to float16 (FP16) if desired for storage, though Parquet handles float32/64 well.
                # df_table2['routing_score'] = df_table2['routing_score'].astype(np.float16)
                
                parquet_file_path = os.path.join("data/routing_scores", f"request_{entry_idx}_routing_scores.parquet")
                df_table2.to_parquet(parquet_file_path, index=False)
        else:
            print(f"Warning: router_logits not found in outputs for entry {entry_idx}. Ensure model.config.output_router_logits=True.")


    # Save Table 1 data
    if all_request_data_table1:
        # Convert it to parquet too
        df_table1 = pd.DataFrame(all_request_data_table1)
        parquet_file_path = os.path.join("data/request_text", "all_request_text_data.parquet")
        df_table1.to_parquet(parquet_file_path, index=False)
    
    print("Processing complete. Data saved to data/")

if __name__ == "__main__":
    main()
