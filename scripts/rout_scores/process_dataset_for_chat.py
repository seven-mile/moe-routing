import os
import torch
from torch.nn.functional import cross_entropy

main_device = 'auto'
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

from datasets import load_dataset

from tqdm import tqdm

torch.random.manual_seed(19260817)

dataset = load_dataset("lmsys/lmsys-chat-1m", split="train")

TOTAL_ENTRIES = 1024

def remove_last_response(entry):
    conversation = entry['conversation']
    # Remove the last response from the conversation
    while len(conversation) > 0 and conversation[-1]['role'] == 'assistant':
        conversation.pop()
    # Update the entry with the modified conversation
    entry['conversation'] = conversation
    return entry

dd = dataset.select(range(TOTAL_ENTRIES)).map(remove_last_response, num_proc=32)
dd.save_to_disk("data/lmsys-chat-1m-no-last-resp")
