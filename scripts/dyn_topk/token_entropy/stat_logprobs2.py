import os
import torch
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from jinja2 import Template
import torch.nn.functional as F

# 确保在运行前已安装所需的库：
# pip install torch transformers tqdm accelerate
# 如果内存不足，可能需要使用 accelerate
# pip install accelerate

# 设置环境变量，指定使用的 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"正在使用设备: {device}")

# 1. 设置文件路径和模型信息
INPUT_FILE_PATH = "data/datasets/ceval_results_with_answer.json"
OUTPUT_FILE_PATH = "data/datasets/ceval_full_ppl_results.json"
MODEL_NAME = "Qwen/Qwen3-30B-A3B"

# 2. 加载模型和分词器
print("正在加载模型和分词器...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).eval()
    print("模型和分词器加载完成。")
except Exception as e:
    print(f"加载模型时出错: {e}")
    exit()

# 3. 准备 Jinja2 模板，用于重建第一阶段的 Prompt
doc_to_text_template = Template("{{question.strip()}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\n<think>")

# 4. 加载之前生成的 JSON 数据
try:
    with open(INPUT_FILE_PATH, "r", encoding="utf-8") as f:
        all_results = json.load(f)
    print(f"已加载 {len(all_results)} 条样本数据。")
except FileNotFoundError:
    print(f"错误: 找不到文件 {INPUT_FILE_PATH}。请确保该文件存在。")
    exit()
except Exception as e:
    print(f"加载文件时出错: {e}")
    exit()

# 5. 主循环：处理每个样本并计算 PPL
ppl_results = []
print("开始计算逐 token PPL...")
for sample in tqdm(all_results, desc="计算 PPL"):
    try:
        # 重建第一阶段的 prompt
        prompt_stage1 = doc_to_text_template.render(sample)
        
        # 逐部分进行分词，不添加特殊 token
        prompt_ids = tokenizer(prompt_stage1, add_special_tokens=False).input_ids
        think_ids = tokenizer(sample['think_part'], add_special_tokens=False).input_ids
        think_end_ids = tokenizer("</think>", add_special_tokens=False).input_ids
        answer_ids = tokenizer(sample['answer_part'], add_special_tokens=False).input_ids
        
        # 拼接所有部分的 token IDs，并手动添加 BOS token
        bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else None
        
        full_ids_list = []
        if bos_token_id is not None:
            full_ids_list.append(bos_token_id)
        
        full_ids_list.extend(prompt_ids)
        full_ids_list.extend(think_ids)
        full_ids_list.extend(think_end_ids)
        full_ids_list.extend(answer_ids)
        
        input_ids = torch.tensor([full_ids_list]).to(device)
        
        # 6. 计算各部分在完整序列中的起始和结束索引
        # 由于我们手动拼接了 ID，长度是精确的
        len_prompt = len(prompt_ids)
        len_think = len(think_ids)
        len_think_end = len(think_end_ids)
        len_answer = len(answer_ids)
        
        # 1. Prompt 部分 (跳过 BOS token)
        prompt_start_idx = 0
        prompt_end_idx = prompt_start_idx + len_prompt
        
        # 2. 思考部分
        think_start_idx = prompt_end_idx
        think_end_idx = think_start_idx + len_think
        
        # 3. 答案部分（包括 </think> 结束符）
        answer_start_idx = think_end_idx
        answer_end_idx = answer_start_idx + len_think_end + len_answer
        
        # 7. 一次前向传播，获取 logits
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits
            
        # 8. 手动计算逐 token 损失 (这里使用 -1 偏移)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        per_token_losses_tensor = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        per_token_losses_tensor = per_token_losses_tensor.view(input_ids.size(0), -1).squeeze(0)
        
        # 转换为 PPL (e^loss)
        per_token_ppls_tensor = torch.exp(per_token_losses_tensor)
        
        # 9. 提取并存储各部分的 PPL 列表
        # 注意: 这里使用 len() 而不是索引，可以避免 -1 偏移带来的复杂性
        prompt_ppls = per_token_ppls_tensor[prompt_start_idx:prompt_end_idx].tolist()
        think_ppls = per_token_ppls_tensor[think_start_idx:think_end_idx].tolist()
        answer_ppls = per_token_ppls_tensor[answer_start_idx:answer_end_idx].tolist()
        
        # 10. 验证长度是否匹配
        total_len = len(prompt_ppls) + len(think_ppls) + len(answer_ppls)
        expected_len = len(per_token_ppls_tensor)
        
        # 理论上这里的警告应该不会再出现
        if total_len != expected_len:
            print(f"\n警告: ID {sample.get('id')} 的 PPL 长度不匹配。实际: {total_len}, 期望: {expected_len}")

        ppl_results.append({
            "id": sample.get("id"),
            "subject": sample.get("subject"),
            "prompt_part_ppls": prompt_ppls,
            "think_part_ppls": think_ppls,
            "answer_part_ppls": answer_ppls
        })
        
    except Exception as e:
        print(f"\n处理样本 {sample.get('id')} (主题: {sample.get('subject')}) 时出错: {e}")
        ppl_results.append({
            "id": sample.get("id"),
            "subject": sample.get("subject"),
            "error": str(e)
        })
        continue

# 11. 保存结果到新的 JSON 文件
print(f"\n正在将结果写入到 {OUTPUT_FILE_PATH}...")
with open(OUTPUT_FILE_PATH, "w", encoding="utf-8") as f:
    json.dump(ppl_results, f, ensure_ascii=False, indent=4)

print("所有任务完成！PPL 结果已成功保存。")
