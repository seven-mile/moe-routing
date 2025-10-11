import os
import torch
from torch.nn.functional import cross_entropy

main_device = 'auto'
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk, load_dataset

from tqdm import tqdm

torch.random.manual_seed(19260817)

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-30B-A3B", device_map=main_device, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B", device_map=main_device)

dataset = load_dataset("lmsys/lmsys-chat-1m", split="train")

full_ppls = torch.load("data/dyn_topk/simulated_bench/qwen3_ppls_32.pt")

TOTAL_ENTRIES = 1

def calc_perplexity(logits, token_ids):
  assert logits.shape[:-1] == token_ids.shape, \
      f"Logits shape {logits.shape} does not match token_ids shape {token_ids.shape}"
  loss = cross_entropy(logits.view(-1, logits.size(-1)), token_ids.view(-1), reduction='none')
  perplexity = torch.exp(loss)
  return perplexity.view(token_ids.shape)

model.eval()
model.config.use_cache = False
model.config.output_logits = True

# for entry in tqdm(dataset.take(TOTAL_ENTRIES).to_iterable_dataset(), total=TOTAL_ENTRIES):
for entry in tqdm(dataset.select(range(TOTAL_ENTRIES))):
  chat = tokenizer.apply_chat_template(entry["conversation"], tokenize=False)
  inputs = tokenizer(chat, return_tensors="pt").to(model.device)

  with torch.no_grad():
    outputs = model(**inputs)

  input_ids = inputs.input_ids[0]
  input_length = input_ids.size(0)
  target_ids = input_ids[1:]

  # Calculate perplexity from logits.
  # Remove the last token to avoid sampling the next token
  output_logits = outputs.logits[0, :-1, :]  # [batch_size, seq_len, vocab_size]
  full_perplexity = calc_perplexity(output_logits, target_ids)
  print(f"Full Perplexity: {full_perplexity.mean().item()}")

  base_top_k = model.config.num_experts_per_tok
  token_top_ks = torch.where(full_perplexity > 1.2, base_top_k, base_top_k-1)

  with torch.no_grad():
    outputs = model(
      **inputs,
      token_top_ks=token_top_ks,
    )

  output_logits = outputs.logits[0, :-1, :]
  reduced_perplexity = calc_perplexity(output_logits, target_ids)
  print(f"Reduced Perplexity: {reduced_perplexity.mean().item()}")
