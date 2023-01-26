# %%
import argparse

import torch

from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("gpt2")
# %%
idx = 0
# %%

# Load the losses
input_ids_meta = torch.load(f"{idx}_input_ids_meta.pt")  # [batch_size, seq_len]
input_ids = torch.load(f"{idx}_input_ids.pt")  # [batch_size, seq_len]

loss_meta = torch.load(f"{idx}_loss_meta.pt")  # [batch_size, seq_len]
loss = torch.load(f"{idx}_loss.pt")  # [batch_size, seq_len]

# %%
# Print the losses
print("Meta")
tok_losses = [
    (tokenizer.decode(input_ids_meta[..., i]), round(loss_meta[..., i].item(), 2))
    for i in range(input_ids_meta.shape[-1] - 1)
]
print(tok_losses[:32])

print("Normal")
tok_losses = [
    (tokenizer.decode(input_ids[..., i]), round(loss[..., i].item(), 2)) for i in range(input_ids.shape[-1] - 1)
]
print(tok_losses[:32])
# %%
