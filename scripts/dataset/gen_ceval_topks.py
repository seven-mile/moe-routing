from configs.ppl_to_ks import spec_default3_mask2025

import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

# --- 1. Configuration ---
# Model, dataset, and generation parameters.
MODEL_NAME = "Qwen/Qwen3-30B-A3B"  # Use a model path compatible with transformers
SPEC_MODEL_NAME = "Qwen/Qwen3-0.6B"  # Model for speculative generation
DATASET_SUBSETS = ['high_school_mathematics', 'computer_architecture']
MAX_SAMPLES = 256  # Limit the number of samples for demonstration
OUTPUT_FILE = "data/dyn_topk/dump_topks/ceval_results_with_top_k.jsonl"

# General generation parameters for assisted top-k
spec_model = AutoModelForCausalLM.from_pretrained(
    SPEC_MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)
GEN_PARAMS_TOPK = {
    "use_cache": True,
    "return_dict_in_generate": True,
    "assistant_model": spec_model,
    "use_assisted_topk": True,
    "num_assistant_tokens": 4,
    "num_assistant_tokens_schedule": 'constant',
    "assisted_action": spec_default3_mask2025,
}

# Generation parameters for the 'think' stage
GEN_PARAMS_THINK = {
    "max_new_tokens": 2048,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "do_sample": True,
}

# Generation parameters for the 'answer' stage
GEN_PARAMS_ANSWER = {
    "max_new_tokens": 2048,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "do_sample": True,
}

# --- 2. Initialization ---
# Load model and tokenizer. Using device_map="auto" for simple multi-GPU handling.
print("Initializing model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
    trust_remote_code=True
)

print("Model and tokenizer initialized.")

# Define stop token IDs. This is more reliable than string-based stopping.
stop_think_id = tokenizer.encode("</think>", add_special_tokens=False)[0]
stop_answer_id = tokenizer.eos_token_id

# --- 3. Data Preparation ---
# Load all specified subsets from the dataset.
all_requests = []
print("Loading and preparing CEVAL dataset...")
for subset in tqdm(DATASET_SUBSETS, desc="Loading subsets"):
    try:
        dataset = load_dataset('ceval/ceval-exam', subset, split='test')
        for doc in dataset:
            doc['subject'] = subset
            all_requests.append(doc)
    except Exception as e:
        print(f"Skipping subset '{subset}' due to an error: {e}")

# Slice the dataset to the desired size.
all_requests = all_requests[:MAX_SAMPLES]
print(f"Dataset loaded. Processing {len(all_requests)} samples.")

# --- 4. Generation and Incremental Saving ---
# Open the output file in append mode to save results one by one.
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for request_data in tqdm(all_requests, desc="Processing requests"):
        # --- Stage 1: Generate the 'think' part ---

        # Format the prompt for the first stage.
        prompt_s1_text = (
            f"{request_data['question'].strip()}\n"
            f"A. {request_data['A']}\nB. {request_data['B']}\n"
            f"C. {request_data['C']}\nD. {request_data['D']}\n<think>"
        )
        
        # Tokenize the prompt.
        prompt_s1_ids = tokenizer.encode(prompt_s1_text, return_tensors="pt").to(model.device)

        # Generate the thinking process.
        outputs_s1 = model.generate(
            prompt_s1_ids,
            **GEN_PARAMS_TOPK,
            **GEN_PARAMS_THINK,
            eos_token_id=stop_think_id,
            output_token_top_ks=True, # Custom flag to get top_k values
            pad_token_id=tokenizer.eos_token_id # Suppress warning
        )

        # Extract generated token IDs and corresponding top_k values.
        generated_s1_ids = outputs_s1.sequences[0][prompt_s1_ids.shape[1]:]
        top_ks_s1 = [k.item() for k in outputs_s1.token_top_ks]

        # --- Stage 2: Generate the final answer ---

        # The input for stage 2 is the full output sequence from stage 1.
        prompt_s2_ids = outputs_s1.sequences

        # Generate the final answer.
        outputs_s2 = model.generate(
            prompt_s2_ids,
            **GEN_PARAMS_TOPK,
            **GEN_PARAMS_ANSWER,
            eos_token_id=stop_answer_id,
            output_token_top_ks=True,
            pad_token_id=tokenizer.eos_token_id
        )

        # Extract the new token IDs and their top_k values.
        generated_s2_ids = outputs_s2.sequences[0][prompt_s2_ids.shape[1]:]
        top_ks_s2 = [k.item() for k in outputs_s2.token_top_ks]

        # --- 5. Save the result for the current request ---
        
        # Structure the data with token IDs and top_k values.
        result = {
            "id": request_data.get('id'),
            "subject": request_data.get('subject'),
            "prompt_ids": prompt_s1_ids[0].tolist(),
            "think_part_ids": generated_s1_ids.tolist(),
            "think_part_top_ks": top_ks_s1,
            "answer_part_ids": generated_s2_ids.tolist(),
            "answer_part_top_ks": top_ks_s2,
        }
        
        # Write the JSON object as a single line to the file.
        f.write(json.dumps(result) + "\n")

print(f"\nAll tasks complete. Results saved to {OUTPUT_FILE}.")

# --- Optional: Verification ---
print("\n--- Example from output file ---")
with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
    first_line = f.readline()
    print(json.dumps(json.loads(first_line), indent=2))
