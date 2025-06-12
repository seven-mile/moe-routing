from transformers import AutoModelForCausalLM, AutoTokenizer

from customizations.deepseek import MoEGate

from datasets import load_dataset, load_from_disk, Dataset, DatasetDict

from tqdm import tqdm

import torch
import torch.nn.functional as F

import pdb
import numpy as np
import pandas as pd
import os

def customize_deepseek_model(model):
    moes_to_patch = []
    for name, module in model.named_modules():
        if name.endswith('.mlp') and hasattr(module, 'gate'):
            moes_to_patch.append(module)

    num_moe_layers = len(moes_to_patch)
    num_experts = model.config.n_routed_experts
    num_truncated_tokens = 2048
    # Buffer for router logits
    router_logits_tensor = torch.empty((num_moe_layers, num_truncated_tokens, num_experts))

    for moe_layer_id, moe in enumerate(moes_to_patch):
        old_gate = moe.gate
        with torch.device(old_gate.weight.device):
            new_gate = MoEGate(model.config, router_logits_tensor[moe_layer_id])
            # Copy all params to the new gate.
            new_gate.load_state_dict(old_gate.state_dict())
            new_gate.eval()
            moe.gate = new_gate

    return router_logits_tensor

def main():
    TOTAL_ENTRIES = 256

    main_device = 'auto'

    # Create directories for output
    os.makedirs("data/request_text", exist_ok=True)
    os.makedirs("data/routing_scores", exist_ok=True)

    # dataset = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)
    # active_data = dataset.select(range(TOTAL_ENTRIES))
    # dd.save_to_disk("data/active_data")

    active_data = load_from_disk('./data/active_data')
    active_data = active_data.select(range(TOTAL_ENTRIES))

    model = AutoModelForCausalLM.from_pretrained("/opt/app/LLM/DeepSeek-R1-awq", device_map=main_device, torch_dtype=torch.float16, attn_implementation="flash_attention_2", trust_remote_code=True)
    model.config.output_router_logits = True
    model.config.use_cache = False # Important for getting all router logits

    tokenizer = AutoTokenizer.from_pretrained("/opt/app/LLM/DeepSeek-R1-awq", trust_remote_code=True)

    # Data storage for Table 1
    all_request_data_table1 = []

    router_logits_buffer = customize_deepseek_model(model)

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

        with torch.no_grad():
            outputs = model(**inputs)
        
        # Wait all writes of router logits buffer to finish
        torch.cuda.synchronize(model.device)
        
        if router_logits_buffer is not None:
            router_logits_tensor = router_logits_buffer[:, :input_length, :]
            
            # Check if the router_logits tensor is empty (e.g., if num_tokens from model output is 0)
            if router_logits_tensor.numel() == 0:
                print(f"Warning: router_logits tensor for entry {entry_idx} is empty. Skipping Parquet generation for this entry.")
            else:
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
