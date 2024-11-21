import time
import json
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainingArguments, Trainer
from torch.utils.data import DataLoader, Dataset
from datetime import datetime
import torch.distributed as dist

# Initialize distributed setup
dist.init_process_group(backend="nccl")

# Get GPU device for this rank
rank = dist.get_rank()
world_size = dist.get_world_size()
device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
torch.cuda.set_device(device)
print(f"[LOG] Rank {rank}/{world_size} using device {device}")

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train a Hugging Face model with DeepSpeed")
parser.add_argument("--model_name", type=str, required=True, help="Name of the model to load (e.g., 'gpt2')")
parser.add_argument("--num_samples", type=int, default=1000, help="Number of synthetic samples in the dataset")
parser.add_argument("--sequence_length", type=int, default=1024, help="Sequence length for the synthetic dataset")
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

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)

# Create synthetic dataset
class RandomTextDataset(Dataset):
    def __init__(self, tokenizer, num_samples, seq_len):
        self.input_ids = torch.randint(0, tokenizer.vocab_size, (num_samples, seq_len), dtype=torch.long)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {"input_ids": self.input_ids[idx], "labels": self.input_ids[idx]}

dataset = RandomTextDataset(tokenizer, args.num_samples, args.sequence_length)

# Define TrainingArguments
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
    dataloader_pin_memory=True,  # Pin memory for efficiency
)

# Custom Trainer
class TokenSpeedTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_samples = 0  # Initialize sample counter

    def training_step(self, model, inputs):
        # Move inputs to the correct device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Perform the standard training step
        output = super().training_step(model, inputs)

        # Update the total samples processed
        batch_size = inputs['input_ids'].size(0)
        # Account for gradient accumulation steps
        self.total_samples += batch_size * self.args.gradient_accumulation_steps

        return output

    def train(self, **kwargs):
        # Ensure that the DistributedSampler is set
        if isinstance(self.train_dataset, torch.utils.data.Dataset):
            self.train_dataloader = None  # Force re-creation of dataloader
        else:
            raise ValueError("train_dataset must be a torch.utils.data.Dataset")

        # Start timing
        start_time = time.time()
        super().train(**kwargs)
        end_time = time.time()

        elapsed_time = end_time - start_time

        # Total tokens processed by this rank
        local_total_tokens = self.total_samples * args.sequence_length

        # Throughput for this rank
        tokens_per_second_rank = local_total_tokens / elapsed_time

        # Aggregate total tokens across all ranks
        total_tokens_tensor = torch.tensor(local_total_tokens).to(device)
        dist.all_reduce(total_tokens_tensor, op=dist.ReduceOp.SUM)
        global_total_tokens = total_tokens_tensor.item()

        # Global throughput
        tokens_per_second_global = global_total_tokens / elapsed_time

        # Average tokens per GPU per second
        avg_tokens_per_gpu = tokens_per_second_global / world_size

        if rank == 0:
            print(f"[LOG] Global Throughput: {tokens_per_second_global:.2f} tokens/second")
            print(f"[LOG] Average Tokens/GPU/Second: {avg_tokens_per_gpu:.2f}")

        return tokens_per_second_rank, tokens_per_second_global

# Instantiate the Trainer
trainer = TokenSpeedTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
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
    print(f"Model Name:             {args.model_name}")
    print(f"Batch Size per GPU:     {args.batch_size}")
    print(f"Gradient Accum Steps:   {args.gradient_accumulation_steps}")
    print(f"Effective Batch Size:   {args.batch_size * args.gradient_accumulation_steps}")
    print(f"Sequence Length:        {args.sequence_length}")
    print(f"Num Samples:            {args.num_samples}")
    print(f"Total GPUs:             {world_size}")
    print(f"Global Throughput:      {tokens_per_second_global:.2f} tokens/second")
    print(f"Average Tokens/GPU/Sec: {tokens_per_second_global / world_size:.2f}")
    print("="*40)
