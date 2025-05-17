from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3_moe import Qwen3MoeForCausalLM
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeDecoderLayer, Qwen3MoeSparseMoeBlock
from datasets import load_dataset

from tqdm import tqdm

import torch
import torch.nn.functional as F

def shapes_to_string(shapes):
    if isinstance(shapes, torch.Tensor):
        return f"{shapes.shape}"
    elif isinstance(shapes, (tuple, list)):
        return f"({', '.join(map(shapes_to_string, shapes))})"
    elif isinstance(shapes, dict):
        return f"{{{', '.join(f'{k}: {shapes_to_string(v)}' for k, v in shapes.items())}}}"
    else:
        assert False, f"Unsupported type: {type(shapes)}"

def patch_model(model: Qwen3MoeForCausalLM):
    handles = []

    def get_hook(lid):
        def hook(module, input, output):
            # print(f"Layer {lid} - Input: {shapes_to_string(input)}, Output: {shapes_to_string(output)}", flush=True)
            # TODO: persist input[0] (1-dim torch.FloatTensor) in SQLite
            pass
        return hook

    for lid, layer in enumerate(model.model.layers):
        layer: Qwen3MoeDecoderLayer
        if not isinstance(layer.mlp, Qwen3MoeSparseMoeBlock):
            continue
        assert isinstance(layer.mlp.gate, torch.nn.Linear), "Expected gate to be a Linear layer"
        handles.append(layer.mlp.gate.register_forward_hook(get_hook(lid)))

    return model, handles

def process_conv(conv):
    # Remove the last response from assistant if it exists
    if conv[-1]['role'] == 'assistant':
        conv = conv[:-1]
    return conv

def main():
    main_device = 'auto'
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-30B-A3B", device_map=main_device)
    model, handles = patch_model(model)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B", device_map=main_device)
    dataset = load_dataset("lmsys/lmsys-chat-1m", split="train")
    
    ppls = []

    TOTAL_ENTRIES = 10

    for entry in tqdm(dataset.take(TOTAL_ENTRIES).to_iterable_dataset(), total=TOTAL_ENTRIES):
        conv = process_conv(entry["conversation"])
        chat = tokenizer.apply_chat_template(conv, tokenize=False)
        inputs = tokenizer(chat, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=4096,
            return_dict_in_generate=True,
            output_logits=True,
            output_hidden_states=True,
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
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1))
        perplexity = torch.exp(loss)

        ppls.append(perplexity.item())
        print('Output Len:', num_generated, 'PPL calc:', ppls[-1], flush=True)

    print(ppls, flush=True)
    print('Average ppl:', sum(ppls) / len(ppls), flush=True)

if __name__ == "__main__":
    main()
