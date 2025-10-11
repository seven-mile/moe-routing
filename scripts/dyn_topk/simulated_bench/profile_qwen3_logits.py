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

  # WARN: This is not well tested.
  # Extract input and output tokens with their IDs and strings
  input_ids = inputs.input_ids[0].tolist()  # Convert to list for the first (and only) batch
  input_tokens = tokenizer.batch_decode(input_ids)
  
  output_ids = outputs.sequences[0, input_length:].tolist()  # Only the generated portion
  output_tokens = tokenizer.batch_decode(output_ids)
  
  # Create structured data entry
  entry_data = {
    'generated_text': result,
    'perplexity': perplexity.tolist(),
    'input_token_ids': input_ids,
    'input_tokens': input_tokens,
    'output_token_ids': output_ids,
    'output_tokens': output_tokens
  }
  
  ppls.append(entry_data)
  print('PPL calc:', len(ppls), 'entries processed', flush=True)

torch.save(ppls, "data/logits/qwen3_ppls_32.pt")
