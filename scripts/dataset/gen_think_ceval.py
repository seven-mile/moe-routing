import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

import torch
import json
from datasets import load_dataset, get_dataset_config_names
from jinja2 import Template
from vllm import LLM, SamplingParams
from tqdm import tqdm

# 1. 准备 Jinja2 模板
# 这个模板将每个样本格式化为模型需要的输入格式
doc_to_text_template = Template("{{question.strip()}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\n<think>")

# 2. 加载 CEVAL 数据集的所有子集 (subjects)
try:
    subset_names = get_dataset_config_names('ceval/ceval-exam')
    print(f"成功发现 {len(subset_names)} 个 CEVAL 子集。")
except Exception as e:
    print(f"无法自动获取子集列表，将使用一个预定义的列表。错误: {e}")
    # 如果无法自动获取，可以使用一个已知的子集列表
    subset_names = [
        'accountant', 'advanced_mathematics', 'art_studies', 'basic_medicine',
        'business_administration', 'chinese_language_and_literature', 'civil_servant',
        'clinical_medicine', 'college_chemistry', 'college_economics', 'college_physics',
        'computer_architecture', 'computer_network', 'education', 'electrical_engineer',
        'environmental_science', 'fire_engineer', 'high_school_biology',
        'high_school_chemistry', 'high_school_chinese', 'high_school_geography',
        'high_school_history', 'high_school_mathematics', 'high_school_physics',
        'high_school_politics', 'ideological_and_moral_cultivation',
        'introduction_to_mao_zedong_thought', 'law', 'legal_professional',
        'logic', 'marxism', 'metrology_engineer', 'middle_school_biology',
        'middle_school_chemistry', 'middle_school_geography',
        'middle_school_history', 'middle_school_mathematics',
        'middle_school_physics', 'middle_school_politics', 'modern_chinese_history',
        'official_document_writing', 'operating_system', 'pharmacy', 'physics',
        'plant_protection', 'probability_and_statistics', 'professional_comprehensive',
        'psychology', 'public_security', 'sports_science', 'tax_accountant',
        'teacher_qualification', 'urban_and_rural_planner', 'veterinary_medicine',
        'world_history'
    ]

# 3. 准备所有待处理的样本
# 我们将同时存储 prompt 和原始问题信息
all_requests = []
print("正在加载和预处理 CEVAL 数据集...")
for subset in tqdm(subset_names, desc="加载子集"):
    try:
        dataset = load_dataset('ceval/ceval-exam', subset, split='test')
        for doc in dataset:
            prompt_text = doc_to_text_template.render(doc)
            # 存储 prompt 和需要记录的原始数据
            all_requests.append({
                "prompt": prompt_text,
                "question_info": {
                    "id": doc.get('id'),
                    "question": doc.get('question'),
                    "subject": subset
                }
            })
    except Exception as e:
        print(f"跳过子集 '{subset}'，因为加载时发生错误: {e}")

print(f"数据集加载完成，共计 {len(all_requests)} 条样本。")

# 提取所有 prompts 用于 VLLM 输入
all_prompts = [req['prompt'] for req in all_requests]

# 4. 初始化 VLLM 模型
# 请将 "your_model_name_or_path" 替换为您要使用的模型名称或本地路径
# 例如: "meta-llama/Llama-2-7b-chat-hf" 或 "/path/to/your/model"
# tensor_parallel_size 可以根据你的 GPU 数量进行调整以加速
print("正在初始化 VLLM 模型...")
llm = LLM(
    model="Qwen/Qwen3-30B-A3B",
    # tensor_parallel_size=torch.cuda.device_count(), # 根据可用 GPU 数量设置
    trust_remote_code=True # 对于某些模型是必需的
)
print("模型初始化完成。")

# 5. 设置采样参数 (SamplingParams)
# For thinking mode, use Temperature=0.6, TopP=0.95, TopK=20, and MinP=0 (the default setting in generation_config.json). DO NOT use greedy decoding, as it can lead to performance degradation and endless repetitions. For more detailed guidance, please refer to the Best Practices section.
sampling_params = SamplingParams(
    max_tokens=4096,
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    min_p=0.0,  # 默认值
    stop=["</think>", "<|end_of_turn|>"]
)

# 6. 使用 VLLM 批量生成结果
print("开始使用 VLLM 进行批量推理...")
outputs = llm.generate(all_prompts, sampling_params)
print("推理完成。")

# 7. 结构化结果并准备写入 JSON
results_to_save = []
print("正在整理输出结果...")
for i, output in enumerate(tqdm(outputs, desc="处理结果")):
    # VLLM 保证输出顺序与输入顺序一致
    original_request = all_requests[i]
    
    # 获取生成的主要文本
    generated_text = output.outputs[0].text
    # 获取停止原因 ('stop', 'length', etc.)
    finish_reason = output.outputs[0].finish_reason

    results_to_save.append({
        "id": original_request["question_info"]["id"],
        "subject": original_request["question_info"]["subject"],
        "question": original_request["question_info"]["question"],
        "think_part": generated_text.strip(), # 移除可能的前后空白
        "stop_reason": finish_reason
    })

# 8. 将结果保存到 JSON 文件
output_file_path = "ceval_results.json"
print(f"正在将结果写入到 {output_file_path}...")
with open(output_file_path, "w", encoding="utf-8") as f:
    json.dump(results_to_save, f, ensure_ascii=False, indent=4)

print("所有任务完成！结果已成功保存。")

# (可选) 打印一两条结果进行验证
print("\n--- 结果示例 ---")
for i in range(min(2, len(results_to_save))):
    print(json.dumps(results_to_save[i], indent=2, ensure_ascii=False))
