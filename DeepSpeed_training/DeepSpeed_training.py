import time
import json
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from transformers import TrainingArguments, Trainer
from datetime import datetime
import torch.distributed as dist

# Debug start
print("[LOG] Script started")

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train a Hugging Face model with DeepSpeed")
parser.add_argument("--model_name", type=str, required=True, help="Name of the model to load (e.g., meta-llama/Llama-3.1-8B)")
parser.add_argument("--num_samples", type=int, default=1000, help="Number of synthetic samples in the dataset")
parser.add_argument("--sequence_length", type=int, default=4096, help="Sequence length for the synthetic dataset")
parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size per GPU")
args, unknown = parser.parse_known_args()

# Generate DeepSpeed configuration
def create_deepspeed_config():
    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "fp16": {"enabled": True},
        "optimizer": {
            "type": "AdamW",
            "params": {"lr": "auto", "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": "auto"}
        },
        "scheduler": {"type": "WarmupLR", "params": {"warmup_min_lr": 0, "warmup_max_lr": "auto", "warmup_num_steps": "auto"}},
        "zero_optimization": {"stage": 2},
    }
    with open("ds_config.json", "w") as f:
        json.dump(ds_config, f, indent=4)

create_deepspeed_config()
dist.init_process_group(backend="nccl")

# Get GPU device for this rank
rank = dist.get_rank()
world_size = dist.get_world_size()
device = torch.device(f"cuda:{rank}")
torch.cuda.set_device(device)

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)

# Dataset and DataLoader
class RandomTextDataset(Dataset):
    def __init__(self, tokenizer, num_samples, seq_len):
        self.input_ids = torch.randint(0, tokenizer.vocab_size, (num_samples, seq_len), dtype=torch.long)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {"input_ids": self.input_ids[idx], "labels": self.input_ids[idx]}

dataset = RandomTextDataset(tokenizer, args.num_samples, args.sequence_length)

# Custom Trainer
class TokenSpeedTrainer(Trainer):
    def train(self, **kwargs):
        start_time = time.time()
        super().train(**kwargs)  # Perform training
        end_time = time.time()

        elapsed_time = end_time - start_time
        total_samples_per_rank = len(self.train_dataset) // world_size
        tokens_per_rank = total_samples_per_rank * args.sequence_length * args.batch_size

        tokens_per_second_rank = tokens_per_rank / elapsed_time
        tokens_tensor = torch.tensor(tokens_per_rank).to(device)

        # Aggregate results
        dist.all_reduce(tokens_tensor, op=dist.ReduceOp.SUM)
        total_tokens = tokens_tensor.item()
        tokens_per_second_global = total_tokens / elapsed_time

        if rank == 0:
            print(f"[LOG] Global Throughput: {tokens_per_second_global:.2f} tokens/second")
            print(f"[LOG] Tokens/GPU/Second: {tokens_per_second_rank:.2f} (RANK 0)")

        return tokens_per_second_rank, tokens_per_second_global

# TrainingArguments and Trainer setup
training_args = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    per_device_train_batch_size=args.batch_size,
    num_train_epochs=1,
    logging_steps=50,
    save_steps=1000000,
    save_total_limit=0,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    deepspeed="./ds_config.json",
    fp16=True,
)

trainer = TokenSpeedTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=None
)

# Start test
if rank == 0:
    print(f"[LOG] Test started at: {datetime.now()}")

tokens_per_second_rank, tokens_per_second_global = trainer.train()

# Summary Report
if rank == 0:
    print("\n" + "="*40)
    print("Summary Report")
    print("="*40)
    print(f"Model Name:            {args.model_name}")
    print(f"Batch Size:            {args.batch_size}")
    print(f"Sequence Length:       {args.sequence_length}")
    print(f"Num Samples:           {args.num_samples}")
    print(f"Gradient Accum Steps:  {args.gradient_accumulation_steps}")
    print(f"Tokens/GPU/Second:     {tokens_per_second_rank:.2f}")
    print(f"Global Throughput:     {tokens_per_second_global:.2f}")
    print("="*40)
