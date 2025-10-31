import os
import torch
import json
from datasets import load_dataset, get_dataset_config_names
from jinja2 import Template
from vllm import LLM, SamplingParams
from vllm.utils.udf import UserDefinedFunctionConfig
from tqdm import tqdm

# 设置环境变量，指定使用的 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"

# 1. 准备 Jinja2 模板
# 这个模板用于生成第一阶段的 prompt
doc_to_text_template = Template("{{question.strip()}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\n<think>")

# 2. 加载 CEVAL 数据集的所有子集 (subjects)
# try:
#     subset_names = get_dataset_config_names('ceval/ceval-exam')
#     print(f"成功发现 {len(subset_names)} 个 CEVAL 子集。")
# except Exception as e:
#     print(f"无法自动获取子集列表，将使用一个预定义的列表。错误: {e}")
#     # 如果无法自动获取，可以使用一个已知的子集列表
#     subset_names = [
#         'accountant', 'advanced_mathematics', 'art_studies', 'basic_medicine',
#         'business_administration', 'chinese_language_and_literature', 'civil_servant',
#         'clinical_medicine', 'college_chemistry', 'college_economics', 'college_physics',
#         'computer_architecture', 'computer_network', 'education', 'electrical_engineer',
#         'environmental_science', 'fire_engineer', 'high_school_biology',
#         'high_school_chemistry', 'high_school_chinese', 'high_school_geography',
#         'high_school_history', 'high_school_mathematics', 'high_school_physics',
#         'high_school_politics', 'ideological_and_moral_cultivation',
#         'introduction_to_mao_zedong_thought', 'law', 'legal_professional',
#         'logic', 'marxism', 'metrology_engineer', 'middle_school_biology',
#         'middle_school_chemistry', 'middle_school_geography',
#         'middle_school_history', 'middle_school_mathematics',
#         'middle_school_physics', 'middle_school_politics', 'modern_chinese_history',
#         'official_document_writing', 'operating_system', 'pharmacy', 'physics',
#         'plant_protection', 'probability_and_statistics', 'professional_comprehensive',
#         'psychology', 'public_security', 'sports_science', 'tax_accountant',
#         'teacher_qualification', 'urban_and_rural_planner', 'veterinary_medicine',
#         'world_history'
#     ]

subset_names = [
    'high_school_mathematics',
    'computer_architecture',
]

# 3. 准备所有待处理的样本
# 我们将存储每个样本的完整信息
all_requests = []
print("正在加载和预处理 CEVAL 数据集...")
for subset in tqdm(subset_names, desc="加载子集"):
    try:
        dataset = load_dataset('ceval/ceval-exam', subset, split='test', num_proc=64)
        for doc in dataset:
            # 为每个样本添加一个 'subject' 字段
            doc['subject'] = subset
            all_requests.append(doc)
    except Exception as e:
        print(f"跳过子集 '{subset}'，因为加载时发生错误: {e}")

print(f"数据集加载完成，共计 {len(all_requests)} 条样本。")

all_requests = all_requests[:256]
print(f"仅使用前 {len(all_requests)} 条样本进行处理。")

# 4. 初始化 VLLM 模型
# 请将 "your_model_name_or_path" 替换为您要使用的模型名称或本地路径
# tensor_parallel_size 可以根据你的 GPU 数量进行调整以加速
print("正在初始化 VLLM 模型...")
llm = LLM(
    model="Qwen/Qwen3-30B-A3B", # 替换为您的模型路径
    tensor_parallel_size=torch.cuda.device_count(), # 根据可用 GPU 数量设置
    speculative_config={"model":"Tengyunw/qwen3_30b_moe_eagle3","num_speculative_tokens":4},
    gpu_memory_utilization=0.8,
    enforce_eager=True,
    trust_remote_code=True, # 对于某些模型是必需的
)
print("模型初始化完成。")

# --- 第一阶段：生成思考过程 (think_part) ---

print("\n--- 开始第一阶段：生成思考过程 ---")
# 准备第一阶段的 prompts
prompts_stage1 = [doc_to_text_template.render(req) for req in all_requests]

# 设置第一阶段的采样参数，在 </think> 处停止
sampling_params_think = SamplingParams.from_optional(
    max_tokens=2048,  # 思考过程的最大长度
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    min_p=0.0,
    stop=["</think>", "<|end_of_turn|>"], # 关键停止符
    dyn_assisted_action_config=UserDefinedFunctionConfig(
        file="configs/ppl_to_ks.py",
        function="spec_default3_mask2025"
    )
)

# 使用 VLLM 批量生成思考过程
think_outputs = llm.generate(prompts_stage1, sampling_params_think)
print("第一阶段推理完成。")

# 将生成的 think_part 添加回原始请求数据中
for i, output in enumerate(think_outputs):
    all_requests[i]['think_part'] = output.outputs[0].text.strip()

# --- 第二阶段：生成最终答案 (answer_part) ---

print("\n--- 开始第二阶段：生成最终答案 ---")
# 准备第二阶段的 prompts
# 格式为: question...options...<think>...think_part...</think>
prompts_stage2 = []
for i, req in enumerate(all_requests):
    prompt_part1 = prompts_stage1[i]
    think_part = req['think_part']
    # 拼接成第二阶段的完整 prompt
    prompts_stage2.append(f"{prompt_part1}{think_part}</think>")

# 设置第二阶段的采样参数，生成最终答案直到结束
sampling_params_answer = SamplingParams.from_optional(
    max_tokens=2048,  # 答案部分的最大长度
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    min_p=0.0,
    stop=["<|end_of_turn|>"], # 使用模型的自然结束符
    dyn_assisted_action_config=UserDefinedFunctionConfig(
        file="configs/ppl_to_ks.py",
        function="spec_default3_mask2025"
    )
)

# 使用 VLLM 批量生成最终答案
answer_outputs = llm.generate(prompts_stage2, sampling_params_answer)
print("第二阶段推理完成。")


# 7. 结构化结果并准备写入 JSON
results_to_save = []
print("\n正在整理最终输出结果...")
for i, request_data in enumerate(tqdm(all_requests, desc="处理结果")):
    # 获取第二阶段生成的答案
    answer_part = answer_outputs[i].outputs[0].text.strip()
    
    results_to_save.append({
        "id": request_data.get('id'),
        "subject": request_data.get('subject'),
        "question": request_data.get('question'),
        "A": request_data.get('A'),
        "B": request_data.get('B'),
        "C": request_data.get('C'),
        "D": request_data.get('D'),
        "think_part": request_data.get('think_part'), # 从第一阶段获取
        "answer_part": answer_part # 从第二阶段获取
    })

# 8. 将结果保存到 JSON 文件
output_file_path = "data/datasets/dyn_topk/ceval_results_with_answer.json"
print(f"正在将结果写入到 {output_file_path}...")
with open(output_file_path, "w", encoding="utf-8") as f:
    json.dump(results_to_save, f, ensure_ascii=False, indent=4)

print("所有任务完成！结果已成功保存。")

# (可选) 打印一两条结果进行验证
print("\n--- 结果示例 ---")
for i in range(min(2, len(results_to_save))):
    print(json.dumps(results_to_save[i], indent=2, ensure_ascii=False))
