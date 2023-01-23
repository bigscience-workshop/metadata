import argparse

import torch
from transformers import AutoTokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--idx",
        type=int,
        default=12,
        help="Index of the loss to plot",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Load the losses
    input_ids_meta = torch.load(f"{args.idx}_input_ids_meta.pt") # [batch_size, seq_len]
    input_ids = torch.load(f"{args.idx}_input_ids.pt") # [batch_size, seq_len]

    loss_meta = torch.load(f"{args.idx}_loss_meta.pt") # [batch_size, seq_len]
    loss = torch.load(f"{args.idx}_loss.pt") # [batch_size, seq_len]

    # Print the losses
    print("Meta")
    tok_losses = [(tokenizer.decode(input_ids_meta[..., i]), round(loss_meta[..., i].item(), 2)) for i in range(input_ids_meta.shape[-1]-1)]
    print(tok_losses)

    print("Normal")
    tok_losses = [(tokenizer.decode(input_ids[..., i]), round(loss[..., i].item(), 2)) for i in range(input_ids.shape[-1]-1)]
    print(tok_losses)


