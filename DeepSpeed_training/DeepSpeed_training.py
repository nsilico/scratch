import time
import json
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from deepspeed import init_distributed
from torch.utils.data import DataLoader, Dataset
from transformers import TrainingArguments, Trainer
from datetime import datetime

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
    default=2, 
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
rank = torch.distributed.get_rank()
device = torch.device(f"cuda:{rank}")
torch.cuda.set_device(device)
print(f"[LOG] Using GPU: {device}")

# Load model and tokenizer
model_name = args.model_name
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

# Assign a padding token (use eos_token as pad_token)
tokenizer.pad_token = tokenizer.eos_token

# Load model and set pad_token_id
model = AutoModelForCausalLM.from_pretrained(model_name)
model.to(device)  # Explicitly move the model to the correct device
if model.config.pad_token_id is None:
    model.config.pad_token_id = tokenizer.pad_token_id

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

# Create dataset
num_samples = args.num_samples
sequence_length = args.sequence_length
batch_size = args.batch_size
dataset = RandomTextDataset(tokenizer, num_samples=num_samples, seq_len=sequence_length)

# Define data collator without pinning
def collate_fn_with_device(batch, device):
    collated_batch = {key: torch.stack([example[key] for example in batch]).to(device, non_blocking=True) for key in batch[0]}
    return collated_batch

# Custom DataLoader
data_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    collate_fn=lambda batch: collate_fn_with_device(batch, device),
)

# Define DeepSpeed configuration
training_args = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    per_device_train_batch_size=batch_size,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=50,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    deepspeed="./ds_config.json",
    fp16=True  # Explicitly enable FP16 in TrainingArguments
)

# Custom Trainer
class TokenSpeedTrainer(Trainer):
    def train(self, **kwargs):
        print("[LOG] Training started...")
        try:
            # Start timing
            start_time = time.time()
            super().train(**kwargs)  # Use Hugging Face's Trainer train method
            end_time = time.time()

            # Calculate throughput
            elapsed_time = end_time - start_time
            total_samples = len(self.train_dataset)  # Total samples in dataset
            sequence_length = args.sequence_length  # Use argument-defined sequence length
            batch_size = self.args.per_device_train_batch_size

            # Total tokens processed
            total_tokens = total_samples * sequence_length * batch_size
            tokens_per_second = total_tokens / elapsed_time

            print(f"[LOG] Training tokens per second: {tokens_per_second:.2f}")
            return tokens_per_second, elapsed_time

        except Exception as e:
            print(f"[LOG] Training failed with exception: {e}")
            raise
        finally:
            print("[LOG] Training completed.")

# Instantiate trainer
trainer = TokenSpeedTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=None  # Use the Trainer's default data collator
)

# Run training and calculate throughput
start_time = datetime.now()
print(f"[LOG] Test started at: {start_time}")
tokens_per_second, duration = trainer.train()
end_time = datetime.now()

# Structured Output
print("\n" + "="*40)
print("Summary Report")
print("="*40)
print(f"Test Start Time:       {start_time}")
print(f"Test End Time:         {end_time}")
print(f"Total Duration:        {duration:.2f} seconds")
print(f"Model Name:            {args.model_name}")
print(f"Batch Size:            {args.batch_size}")
print(f"Sequence Length:       {args.sequence_length}")
print(f"Num Samples:           {args.num_samples}")
print(f"Gradient Accum Steps:  {args.gradient_accumulation_steps}")
print(f"Tokens/Second:         {tokens_per_second:.2f}")
print("="*40)
