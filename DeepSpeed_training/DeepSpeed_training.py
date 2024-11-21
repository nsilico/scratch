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
parser.add_argument(
    "--model_name", 
    type=str, 
    required=True, 
    help="Name of the model to load from Hugging Face (e.g., meta-llama/Llama-3.1-8B)"
)
parser.add_argument(
    "--num_samples", 
    type=int, 
    default=1000, 
    help="Number of synthetic samples in the dataset"
)
parser.add_argument(
    "--sequence_length", 
    type=int, 
    default=4096, 
    help="Sequence length for the synthetic dataset"
)
parser.add_argument(
    "--gradient_accumulation_steps", 
    type=int, 
    default=1, 
    help="Number of steps to accumulate gradients before performing an optimizer step"
)
parser.add_argument(
    "--batch_size", 
    type=int, 
    default=1, 
    help="Per-device batch size for training"
)
try:
    args, unknown = parser.parse_known_args()
    print(f"[LOG] Parsed arguments: {args}")
    print(f"[LOG] Unknown arguments passed to script: {unknown}")
except SystemExit as e:
    print("[LOG] Error parsing arguments. Ensure arguments are provided correctly.")
    raise

# Initialize distributed environment
def init_distributed():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size

# Initialize distributed training
rank, world_size = init_distributed()
device = torch.device(f"cuda:{rank}")
torch.cuda.set_device(device)
if rank == 0:
    print(f"[LOG] Using GPU: {device} on rank 0")
print(f"[LOG] Rank {rank}/{world_size} initialized.")

# Generate synthetic dataset
class RandomTextDataset(Dataset):
    def __init__(self, tokenizer, num_samples=100, seq_len=4096):
        self.input_ids = torch.randint(
            0, tokenizer.vocab_size, (num_samples, seq_len), dtype=torch.long
        )

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {"input_ids": self.input_ids[idx], "labels": self.input_ids[idx]}

# DistributedSampler ensures each rank processes different parts of the dataset
from torch.utils.data.distributed import DistributedSampler
num_samples = args.num_samples
sequence_length = args.sequence_length
dataset = RandomTextDataset(AutoTokenizer.from_pretrained(args.model_name), num_samples=num_samples, seq_len=sequence_length)
sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)

# Define data collator without pinning GPU tensors
def collate_fn_with_device(batch):
    return {
        key: torch.stack([example[key] for example in batch]).to(device, non_blocking=True)
        for key in batch[0]
    }

# DataLoader with DistributedSampler (no pin_memory)
data_loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    sampler=sampler,
    collate_fn=collate_fn_with_device,
    num_workers=4  # Keep worker count for parallel data loading
)

# Custom Trainer to Track Tokens
class TokenSpeedTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.local_total_samples = 0  # Track samples processed by this rank

    def training_step(self, model, inputs):
        # Perform a standard training step
        output = super().training_step(model, inputs)

        # Increment sample count for this rank
        self.local_total_samples += inputs["input_ids"].size(0)
        return output

    def train(self, **kwargs):
        try:
            # Start timing
            start_time = time.time()
            super().train(**kwargs)  # Perform training
            end_time = time.time()

            # Calculate throughput for this rank
            elapsed_time = end_time - start_time
            local_tokens_processed = self.local_total_samples * sequence_length
            local_tokens_per_second = local_tokens_processed / elapsed_time

            print(f"[RANK {rank}] Tokens per second: {local_tokens_per_second:.2f}")

            # Aggregate tokens across all ranks
            total_tokens_tensor = torch.tensor(local_tokens_processed).to(device)
            dist.all_reduce(total_tokens_tensor, op=dist.ReduceOp.SUM)
            total_tokens_processed = total_tokens_tensor.item()

            # Calculate global throughput
            total_throughput = total_tokens_processed / elapsed_time
            tokens_per_gpu_per_second = total_throughput / world_size

            if rank == 0:
                print(f"[LOG] Average Tokens/GPU/Second: {tokens_per_gpu_per_second:.2f}")
                print(f"[LOG] Total Throughput: {total_throughput:.2f} Tokens/Second")

            return local_tokens_per_second, elapsed_time, tokens_per_gpu_per_second, total_throughput

        except Exception as e:
            print(f"[RANK {rank}] Training failed with exception: {e}")
            raise

# Trainer instantiation
training_args = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    per_device_train_batch_size=args.batch_size,
    num_train_epochs=1,
    logging_steps=50,
    save_steps=1000000,
    save_total_limit=0,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    fp16=True
)
model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
trainer = TokenSpeedTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collate_fn_with_device
)

# Start Training
if rank == 0:
    print(f"[LOG] Test started at: {datetime.now()}")

tokens_per_second, duration, tokens_per_gpu_per_second, total_throughput = trainer.train()

# Log Final Summary
dist.barrier()
if rank == 0:
    print("\n" + "="*40)
    print("Summary Report")
    print("="*40)
    print(f"Model Name:            {args.model_name}")
    print(f"Batch Size:            {args.batch_size}")
    print(f"Sequence Length:       {args.sequence_length}")
    print(f"Num Samples:           {args.num_samples}")
    print(f"Gradient Accum Steps:  {args.gradient_accumulation_steps}")
    print(f"Total GPUs:            {world_size}")
    print(f"Average Tokens/GPU/Second: {tokens_per_gpu_per_second:.2f}")
    print(f"Total Throughput:      {total_throughput:.2f} Tokens/Second")
    print("="*40)
