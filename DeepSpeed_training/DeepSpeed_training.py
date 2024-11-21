import time
import json
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from deepspeed import init_distributed
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
    help="Name of the model to load from Hugging Face (e.g., 'gpt2')"
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
    default=1024, 
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

# Generate DeepSpeed configuration
def create_deepspeed_config():
    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": "auto",
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": "auto"
            }
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": "auto",
                "warmup_num_steps": "auto"
            }
        },
        "zero_optimization": {
            "stage": 2,
            "overlap_comm": True,
            "reduce_scatter": True,
            "contiguous_gradients": True
        },
        "activation_checkpointing": {
            "partition_activations": True,
            "contiguous_memory_optimization": True,
            "cpu_checkpointing": False
        }
    }
    with open("ds_config.json", "w") as f:
        json.dump(ds_config, f, indent=4)
    print("[LOG] DeepSpeed config file created as 'ds_config.json'.")

# Create DeepSpeed config
create_deepspeed_config()

# Initialize distributed setup for DeepSpeed
init_distributed()

# Determine the current rank and assign the corresponding GPU
rank = dist.get_rank()
world_size = dist.get_world_size()
device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
torch.cuda.set_device(device)
if rank == 0:
    print(f"[LOG] Using GPU: {device} on rank 0")
print(f"[LOG] Rank {rank}/{world_size} initialized.")

# Load model and tokenizer
model_name = args.model_name
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

# Assign a padding token (use eos_token as pad_token)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model and set pad_token_id
model = AutoModelForCausalLM.from_pretrained(model_name)
model.to(device)  # Explicitly move the model to the correct device
if model.config.pad_token_id is None:
    model.config.pad_token_id = tokenizer.pad_token_id

# Generate synthetic dataset
class RandomTextDataset(Dataset):
    def __init__(self, tokenizer, num_samples=100, seq_len=1024):
        self.input_ids = torch.randint(
            low=0, high=tokenizer.vocab_size, size=(num_samples, seq_len), dtype=torch.long
        )

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {"input_ids": self.input_ids[idx], "labels": self.input_ids[idx]}

# Create dataset
num_samples = args.num_samples
sequence_length = args.sequence_length
batch_size = args.batch_size
dataset = RandomTextDataset(tokenizer, num_samples=num_samples, seq_len=sequence_length)

# Use a DistributedSampler to ensure each rank gets a different subset of data
from torch.utils.data.distributed import DistributedSampler
sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)

# Define data collator
def collate_fn_with_device(batch):
    collated_batch = {key: torch.stack([example[key] for example in batch]).to(device, non_blocking=True) for key in batch[0]}
    return collated_batch

# Custom DataLoader
data_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=sampler,
    collate_fn=collate_fn_with_device,
    num_workers=4,
    pin_memory=True
)

# Define DeepSpeed configuration
training_args = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    per_device_train_batch_size=batch_size,
    num_train_epochs=1,
    logging_steps=50,
    save_steps=1000000,
    save_total_limit=0,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    deepspeed="./ds_config.json",
    fp16=True
)

# Custom Trainer
class TokenSpeedTrainer(Trainer):
    def __init__(self, *args, data_loader=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_loader = data_loader
        self.total_samples = 0  # Initialize sample counter

    def get_train_dataloader(self):
        if self.data_loader is not None:
            return self.data_loader
        else:
            return super().get_train_dataloader()

    def training_step(self, model, inputs):
        # Perform the standard training step
        output = super().training_step(model, inputs)

        # Update the total samples processed
        batch_size = inputs['input_ids'].size(0)
        self.total_samples += batch_size

        return output

    def train(self, **kwargs):
        try:
            # Start timing
            start_time = time.time()
            super().train(**kwargs)
            end_time = time.time()

            # Calculate throughput
            elapsed_time = end_time - start_time

            # Total tokens processed per rank
            sequence_length = args.sequence_length
            local_total_tokens = self.total_samples * sequence_length

            # Each rank reports its own throughput
            tokens_per_second = local_total_tokens / elapsed_time
            print(f"[RANK {rank}] Tokens per second: {tokens_per_second:.2f}")

            # Aggregate tokens across all ranks
            total_tokens_tensor = torch.tensor(local_total_tokens).to(device)
            dist.all_reduce(total_tokens_tensor, op=dist.ReduceOp.SUM)
            global_total_tokens = total_tokens_tensor.item()

            # Compute total tokens per second
            total_tokens_per_second = global_total_tokens / elapsed_time

            # Compute average tokens per GPU per second
            avg_tokens_per_gpu_per_second = total_tokens_per_second / world_size

            if rank == 0:
                print(f"[LOG] Total Tokens/Second: {total_tokens_per_second:.2f}")
                print(f"[LOG] Average Tokens/GPU/Second: {avg_tokens_per_gpu_per_second:.2f}")

            return tokens_per_second, elapsed_time, total_tokens_per_second, avg_tokens_per_gpu_per_second

        except Exception as e:
            print(f"[RANK {rank}] Training failed with exception: {e}")
            raise

# Instantiate trainer with the custom data loader
trainer = TokenSpeedTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=collate_fn_with_device,
    data_loader=data_loader  # Pass it to our custom trainer
)

# Start test
if rank == 0:
    print(f"[LOG] Test started at: {datetime.now()}")

tokens_per_second, duration, total_tokens_per_second, avg_tokens_per_gpu_per_second = trainer.train()

# Gather and print metrics
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
    print(f"Total Tokens/Second:   {total_tokens_per_second:.2f}")
    print(f"Tokens/GPU/Second:     {avg_tokens_per_gpu_per_second:.2f}")
    print("="*40)
