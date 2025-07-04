import os
import torch
from torch.nn.functional import cross_entropy

main_device = 'auto'
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk

from tqdm import tqdm

torch.random.manual_seed(19260817)

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-30B-A3B", device_map=main_device)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B", device_map=main_device)

dataset = load_from_disk("data/lmsys-chat-1m-no-last-resp")

TOTAL_ENTRIES = 32

ppls = []

# for entry in tqdm(dataset.take(TOTAL_ENTRIES).to_iterable_dataset(), total=TOTAL_ENTRIES):
for entry in tqdm(dataset.select(range(TOTAL_ENTRIES))):
  chat = tokenizer.apply_chat_template(entry["conversation"], tokenize=False)
  inputs = tokenizer(chat, return_tensors="pt").to(model.device)
  outputs = model.generate(
    **inputs,
    max_new_tokens=4096,
    return_dict_in_generate=True,
    output_logits=True,
    # output_hidden_states=True,
  )
  result = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
  # print(f"Stop reason: {outputs.stop_reason}")

  # Decode and print the generated text.
  # print(result)

  # Calculate perplexity from logits.
  logits = torch.stack(outputs.logits, dim=1)  # [batch_size, seq_len, vocab_size]

  # Verify the dimensions of the logits tensor.
  input_length = inputs.input_ids.size(1)
  num_generated = len(outputs.logits)
  assert outputs.sequences.size(1) >= input_length + num_generated, \
      f"Sequence too short: expected at least {input_length + num_generated}, got {outputs.sequences.size(1)}"

  target_ids = outputs.sequences[:, -logits.size(1):]  # Only use the generated portion
  loss = cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1), reduction='none')
  perplexity = torch.exp(loss)

  ppls.append((result, perplexity.tolist()))
  print('PPL calc:', ppls[-1], flush=True)

torch.save(ppls, "data/logits/qwen3_ppls_32.pt")
