import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import torch
import torch.nn.functional as F
import math
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# --- 1. 配置与模型加载 ---
# 为了方便演示和在大多数消费级GPU上运行，我们选择了一个较小但能力很强的模型。
# 您可以根据需要替换成更大的模型，如 Qwen2-7B-Instruct 等。
MODEL_NAME = "Qwen/Qwen3-30B-A3B"
# 我们将使用 'val' 数据集中的一个子集进行演示。
# 若要评估完整数据集，请移除 .select(range(20))
DATASET_NAME = "ceval/ceval-exam"
DATASET_CONFIG = "computer_architecture"
DATASET_SPLIT = "val"
NUM_SAMPLES = None # 演示样本数，设为 None 可运行全集

# 生成参数
MAX_THINK_TOKENS = 4096  # 思考过程最大 token 数
STEP_TOKENS = 4          # 每步生成的 token 数

print("="*20)
print(f"模型: {MODEL_NAME}")
print(f"数据集: {DATASET_NAME} ({DATASET_CONFIG})")
print("="*20 + "\n")

# 加载分词器和模型
# 使用 bfloat16 以提高性能，并使用 device_map="auto" 自动分配到 GPU
print("正在加载模型和分词器...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2",
)
# 将 pad_token_id 设置为 eos_token_id 以避免警告
model.generation_config.pad_token_id = tokenizer.eos_token_id
model.generation_config.do_sample = False
model.generation_config.temperature = None
model.generation_config.top_p = None
model.generation_config.top_k = None
print("模型加载完成。\n")


# --- 2. 加载数据集 ---
print("正在加载数据集...")
if NUM_SAMPLES:
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT).select(range(NUM_SAMPLES))
else:
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT)
print(f"数据集加载完成，共 {len(dataset)} 个样本。\n")


# --- 3. 生成与评估 ---
correct_predictions = 0
total_predictions = 0
options = ["A", "B", "C", "D"]

# 定义思考过程的结束标记
think_stop_string = "</think>"

# 遍历数据集
for doc in tqdm(dataset, desc="正在评估"):
    # a. 构建初始提示 (到 <think> 标签为止)
    prompt_start = (
        f"{doc['question'].strip()}\n"
        f"A. {doc['A']}\n"
        f"B. {doc['B']}\n"
        f"C. {doc['C']}\n"
        f"D. {doc['D']}\n"
        f"<think>"
    )
    input_ids = tokenizer(prompt_start, return_tensors="pt").input_ids.to(model.device)

    # 初始化KV Cache和思考过程的token列表
    past_key_values = None
    generated_think_ids = []
    current_step_input_ids = input_ids

    # b. 分步生成 "思考" 过程
    # 循环直到达到上限或生成停止符
    for _ in tqdm(range(math.ceil(MAX_THINK_TOKENS / STEP_TOKENS)), desc="生成思考过程", leave=False):
        with torch.no_grad():
            outputs = model.generate(
                current_step_input_ids,
                max_new_tokens=STEP_TOKENS,
                use_cache=True,
                past_key_values=past_key_values,
                return_dict_in_generate=True,
                # --- 修改点: 为 "思考" 过程开关采样 ---
                do_sample=False,
                # temperature=0.7,
                # top_p=0.8,
                # top_k=20
            )

        # 提取新生成的 token
        # `outputs.sequences` 包含了输入的token，所以我们只取最后新生成的部分
        new_token_ids = outputs.sequences[0, current_step_input_ids.shape[-1]:]

        # 更新KV Cache以备下一步使用
        past_key_values = outputs.past_key_values

        # 将新生成的token添加到思考过程中
        generated_think_ids.extend(new_token_ids.tolist())

        # 下一步的输入必须是完整句子
        current_step_input_ids = torch.cat([
            current_step_input_ids,
            new_token_ids.unsqueeze(0)
        ], dim=1)

        # 检查是否已生成结束标记
        decoded_think_part = tokenizer.decode(generated_think_ids)
        if think_stop_string in decoded_think_part:
            # 如果生成了</think>，就清理并截断多余的部分
            # 例如，如果模型生成了 "</think>好的，答案是"，我们只保留到"</think>"为止
            # 重新编码以确保 token 序列的准确性
            clean_think_part = decoded_think_part.split(think_stop_string)[0] + think_stop_string
            generated_think_ids = tokenizer.encode(clean_think_part, add_special_tokens=False)
            break
    else:
        clean_think_part = decoded_think_part + think_stop_string
    
    # c. 准备最终的提示
    prompt_end = "\n答案："
    full_sequence_text = prompt_start + clean_think_part + prompt_end
    full_sequence_ids = tokenizer(full_sequence_text, return_tensors="pt").input_ids.to(model.device)

    # --- 修改点: d. LogLikelihood 评测 ---
    # 我们不再生成答案，而是直接获取模型在当前上下文后，对下一个 token 的预测 logits
    with torch.no_grad():
        # 直接进行一次前向传播，而不是 .generate()
        outputs = model(full_sequence_ids)
        
        # outputs.logits 的形状是 [batch_size, sequence_length, vocab_size]
        # 我们需要的是在整个序列末尾的那个 token 的 logits
        last_token_logits = outputs.logits[0, -1, :]

    # 使用 log_softmax 将 logits 转换为对数概率，这在数值上更稳定
    log_probs = F.log_softmax(last_token_logits, dim=-1)

    # --- 修改点: e. 提取并比较各选项的对数似然值 ---
    # 从完整的对数概率分布中，只抽取我们关心的 A, B, C, D 选项的 token 的值
    option_log_likelihoods = {
        option: log_probs[tokenizer.encode(option)].item()
        for option in options
    }
    
    # 选择对数似然值最高的选项作为模型的预测答案
    predicted_answer = max(option_log_likelihoods, key=option_log_likelihoods.get)
    ground_truth_answer = doc["answer"]

    # 打印单项的详细结果（可选）
    print("\n---")
    print(full_sequence_text)
    # 打印每个选项的对数似然值，便于分析
    print(f"LogLikelihoods: {option_log_likelihoods}")
    print(f"模型预测: {predicted_answer}, 正确答案: {ground_truth_answer}")
    print("---\n")

    if predicted_answer == ground_truth_answer:
        correct_predictions += 1
    total_predictions += 1

# --- 4. 输出最终结果 ---
accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
print("\n" + "="*20)
print("LogLikelihood 评估完成！")
print(f"总样本数: {total_predictions}")
print(f"正确数: {correct_predictions}")
print(f"准确率: {accuracy:.2f}%")
print("="*20)
