import os
import json
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from jinja2 import Template
from vllm import LLM, SamplingParams
from tqdm import tqdm

# --- 1. Configuration ---
# Set the visible CUDA devices
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

# Define the model to be used (should match the generation script)
MODEL_NAME = "Qwen/Qwen3-30B-A3B"
# Path to the input JSON file created by the first script
INPUT_JSON_PATH = "data/assisted_dynamic_topk/ceval/ceval_results.json"
# Define the quantiles (percentiles) to calculate.
# Default: 11 deciles (0%, 10%, ..., 100%)
QUANTILES_TO_CALCULATE = list(range(0, 101, 10))


# --- 2. Prepare Jinja2 Template ---
# This template formats each record for the log-probability calculation pass.
prefill_template = Template("{{question.strip()}}<think>{{think_part.strip()}}</think>")


# --- 3. Load Previously Generated Results ---
print(f"正在从 '{INPUT_JSON_PATH}' 加载结果...")
try:
    with open(INPUT_JSON_PATH, "r", encoding="utf-8") as f:
        ceval_results = json.load(f)
    print(f"成功加载 {len(ceval_results)} 条记录。")
except FileNotFoundError:
    print(f"错误: 输入文件 '{INPUT_JSON_PATH}' 未找到。请先运行第一个脚本生成结果。")
    exit()
except json.JSONDecodeError:
    print(f"错误: 无法解析 '{INPUT_JSON_PATH}'。文件可能已损坏或格式不正确。")
    exit()


# --- 4. Initialize VLLM Model and Tokenizer ---
print(f"正在初始化 VLLM 模型: {MODEL_NAME}...")
# Use a context manager to ensure the LLM object is properly handled
try:
    llm = LLM(
        model=MODEL_NAME,
        trust_remote_code=True,
        # tensor_parallel_size can be set here if needed, e.g., torch.cuda.device_count()
        tensor_parallel_size=torch.cuda.device_count(),
        # limit batch size to avoid OOM
        max_num_seqs=1,
    )
    tokenizer = llm.get_tokenizer()
    print("模型和分词器初始化完成。")
except Exception as e:
    print(f"模型初始化失败: {e}")
    exit()


# --- 5. Prepare All Prompts for Prefill and Calculate Prefix Lengths ---
all_requests = []
print("正在准备用于 LogProb 计算的 Prompts...")
for record in tqdm(ceval_results, desc="处理记录"):
    # Some 'think_part' might be empty, ensure it's a string
    think_part = record.get("think_part", "") or ""

    # Construct the full prompt for VLLM
    prompt_text = prefill_template.render(
        question=record["question"],
        think_part=think_part
    )

    # Construct the prefix part to calculate its token length
    # This helps us slice the logprobs list later to isolate the 'think_part'
    prefix_text = Template("{{question.strip()}}<think>").render(question=record["question"])
    prefix_token_ids = tokenizer.encode(prefix_text)
    
    # The '</think>' at the end adds one token that we also want to ignore.
    # We will slice from `len(prefix_token_ids)` to `-1`.
    
    all_requests.append({
        "prompt": prompt_text,
        "prefix_token_count": len(prefix_token_ids),
        "original_info": {
            "id": record.get("id"),
            "subject": record.get("subject"),
        }
    })

all_prompts = [req['prompt'] for req in all_requests]
print(f"共准备了 {len(all_prompts)} 条 prompts。")


# --- 6. Set SamplingParams for LogProb Calculation ---
# To get logprobs for the prompt, we set max_tokens=0 and request logprobs.
sampling_params = SamplingParams(
    max_tokens=1,    # Crucial: Only perform prefill, do not generate new tokens.
    logprobs=1,      # Request log probabilities for the prompt tokens.
    prompt_logprobs=1 # Ensure prompt logprobs are returned
)

# --- 7. Use VLLM to Get Log Probs ---
print("开始使用 VLLM 进行批量 LogProb 计算 (Prefill)...")
outputs = llm.generate(all_prompts, sampling_params)
print("LogProb 计算完成。")


# --- 8. Process Outputs, Calculate Quantiles, and Structure for Pandas ---
final_results = []
print("正在整理输出并计算分位数...")
for i, output in enumerate(tqdm(outputs, desc="处理结果")):
    original_request = all_requests[i]
    prompt_logprobs = output.prompt_logprobs

    # The logprobs for the tokens in the <think>...</think> block
    think_logprobs = []

    if prompt_logprobs:
        prefix_len = original_request["prefix_token_count"]
        # Slice the list to get logprobs only for the {think_part}.
        # We slice from [prefix_len] up to the second to last element [-1]
        # to exclude the logprob of the final '</think>' token.
        target_logprobs_list = prompt_logprobs[prefix_len:-1]
        
        # Extract the float value from each logprob dictionary
        think_logprobs = [list(logprob.values())[0] for logprob in target_logprobs_list if logprob]

    # Prepare a dictionary for the current record's results
    result_dict = {
        "id": original_request["original_info"]["id"],
        "subject": original_request["original_info"]["subject"],
    }

    if think_logprobs:
        # Calculate all requested quantiles at once
        quantiles = np.percentile(think_logprobs, QUANTILES_TO_CALCULATE)
        for q_val, res in zip(QUANTILES_TO_CALCULATE, quantiles):
            result_dict[f"logprob_p{q_val}"] = res
    else:
        # If no logprobs were found, fill with None or NaN
        for q_val in QUANTILES_TO_CALCULATE:
            result_dict[f"logprob_p{q_val}"] = np.nan
            
    final_results.append(result_dict)


# --- 9. Create and Display Pandas DataFrame ---
print("\n所有任务完成！正在将结果转换为 Pandas DataFrame。")
df = pd.DataFrame(final_results)

# Set display options for better viewing
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 120)

print("\n--- 输出结果示例 (前5行) ---")
print(df.head())

# You can now easily save this DataFrame to a CSV or other formats
output_csv_path = "data/assisted_dynamic_topk/ceval/ceval_logprob_quantiles.csv"
print(f"\n将 DataFrame 保存到 '{output_csv_path}'...")
df.to_csv(output_csv_path, index=False)
print("成功保存。")
