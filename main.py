from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3_moe import Qwen3MoeForCausalLM
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeDecoderLayer, Qwen3MoeSparseMoeBlock

from datasets import load_dataset
import pandas as pd

from tqdm import tqdm

import torch
import torch.nn.functional as F

import pdb

def shapes_to_string(shapes):
    if isinstance(shapes, torch.Tensor):
        return f"{shapes.shape}"
    elif isinstance(shapes, (tuple, list)):
        return f"({', '.join(map(shapes_to_string, shapes))})"
    elif isinstance(shapes, dict):
        return f"{{{', '.join(f'{k}: {shapes_to_string(v)}' for k, v in shapes.items())}}}"
    else:
        assert False, f"Unsupported type: {type(shapes)}"

def remove_conv_response(entry):
    # Remove the last response from assistant if it exists
    if entry['conversation'][-1]['role'] == 'assistant':
        entry['conversation'].pop()
    return entry

def patch_model(model: Qwen3MoeForCausalLM, output_router_logits: list):
    handles = []

    def get_hook(lid: int):
        def hook(module, input, output):
            print(f"Layer {lid} - Input: {shapes_to_string(input)}, Output: {shapes_to_string(output)}", flush=True)
            # output: (batch_size * seq_len, num_experts)
            assert output.ndim == 2, f"Expected output to be 2D tensor, got {output.ndim}D"
            # Should collect all seq parts. e.g. 512(prefill), 1, 1, ... (decode)
            output_router_logits[lid].append(output)
        return hook

    for lid, layer in enumerate(model.model.layers):
        layer: Qwen3MoeDecoderLayer
        if not isinstance(layer.mlp, Qwen3MoeSparseMoeBlock):
            continue
        assert isinstance(layer.mlp.gate, torch.nn.Linear), "Expected gate to be a Linear layer"
        output_router_logits.append([])
        handles.append(layer.mlp.gate.register_forward_hook(get_hook(lid)))

    return model, handles

def main():
    BATCH_SIZE = 32
    TOTAL_ENTRIES = 2 # Keep it small for testing, can be increased
    dataset = load_dataset("lmsys/lmsys-chat-1m", split="train")

    active_data = dataset.map(remove_conv_response, num_proc=32) \
        .filter(lambda x: len(x['conversation']) > 0, num_proc=32) \
        .batch(BATCH_SIZE, num_proc=32) \
        .take(TOTAL_ENTRIES)

    main_device = 'cuda:0'
    # It's good practice to specify torch_dtype for large models
    # For A100/H100, bfloat16 is good. For older GPUs, float16.
    # If 'auto' device_map might put parts on CPU, ensure dtype compatibility or manage manually.
    try:
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-30B-A3B", device_map=main_device, torch_dtype=torch.bfloat16, trust_remote_code=True)
    except Exception as e:
        print(f"Failed to load model with bfloat16, trying float16: {e}")
        try:
            model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-30B-A3B", device_map=main_device, torch_dtype=torch.float16, trust_remote_code=True)
        except Exception as e2:
            print(f"Failed to load model with float16, trying default dtype: {e2}")
            model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-30B-A3B", device_map=main_device, trust_remote_code=True)

    output_router_logits = []

    model, handles = patch_model(model, output_router_logits)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B", trust_remote_code=True)
    
    ppls = []
    all_gating_data_rows = [] # To store data for the DataFrame

    for entry_idx, entry in tqdm(enumerate(active_data), total=TOTAL_ENTRIES, desc="Processing Entries"):
        conv = entry['conversation']
        this_batch_size = len(conv)
        chat_string = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)

        inputs = tokenizer(chat_string, padding=True, padding_side='left', return_tensors="pt").to(model.device)

        input_length = inputs.input_ids.size(1)

        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            return_dict_in_generate=True,
            output_logits=False, # For perplexity
            output_hidden_states=False, # Not strictly needed for this task
        )

        # Process and store gating scores
        
        num_layers_to_process = len(output_router_logits)

        for layer_idx in range(num_layers_to_process):
            # Shape: (bseq, num_experts)
            router_scores_list = output_router_logits[layer_idx]
            num_experts = router_scores_list[0].size(-1)

            # Shape: (bs, seqlen, num_experts)
            router_scores = torch.cat([
                x.reshape(this_batch_size, -1, num_experts)
                for x in router_scores_list], dim=1)
            router_scores_list.clear()

            for batch_idx in range(this_batch_size):
                scores_tensor_for_seq = router_scores[batch_idx].cpu().float() # Shape: (full_seq_len, num_experts), ensure float for pd
                full_token_ids = outputs.sequences[batch_idx].cpu() # (full_seq_len)
                assert full_token_ids.size(0) == scores_tensor_for_seq.size(0) + 1, f"Mismatch in token IDs and scores length: {full_token_ids.size(0)} vs {scores_tensor_for_seq.size(0) + 1}"

                for token_pos_in_seq in range(scores_tensor_for_seq.size(0)):
                    is_prompt = token_pos_in_seq < input_length
                    token_id_val = full_token_ids[token_pos_in_seq].item()
                    
                    # Decode individual tokens. skip_special_tokens=False to see all tokens.
                    # clean_up_tokenization_spaces=False to preserve tokenization artifacts like leading spaces.
                    token_str = tokenizer.decode([token_id_val], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    gating_scores_for_token = scores_tensor_for_seq[token_pos_in_seq].numpy()
                    
                    is_special = token_id_val in tokenizer.all_special_ids

                    row_data = {
                        "entry_idx": entry_idx,
                        "batch_idx": batch_idx,
                        "token_pos_in_full_sequence": token_pos_in_seq,
                        "token_id": token_id_val,
                        "token_str": token_str,
                        "is_generated_token": not is_prompt,
                        "is_special_token": is_special,
                        "model_layer_idx": layer_idx,
                        "expert_scores": gating_scores_for_token,
                    }
                    
                    all_gating_data_rows.append(row_data)


    # After the loop, create DataFrame and save to CSV
    if all_gating_data_rows:
        gating_df = pd.DataFrame(all_gating_data_rows)
        try:
            gating_df.to_pickle("./data/gating_scores_and_tokens.pkl")
            print("\nSuccessfully exported gating scores and token data")
        except Exception as e:
            print(f"\nError exporting DataFrame to CSV: {e}")
    else:
        print("\nNo gating data was collected. CSV file not created.")

    if ppls:
        print("\nPerplexities:", ppls, flush=True)
        print('Average ppl:', sum(ppls) / len(ppls), flush=True)
    else:
        print("\nNo perplexity scores were calculated.", flush=True)

if __name__ == "__main__":
    main()
