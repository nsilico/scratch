import time
import json
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from deepspeed import init_distributed
from torch.utils.data import DataLoader, Dataset
from transformers import TrainingArguments, Trainer

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train a Llama model with DeepSpeed")
parser.add_argument(
    "--model_name", 
    type=str, 
    required=True, 
    help="Name of the model to load from Hugging Face (e.g., meta-llama/Llama-3.1-8b-hf)"
)
try:
    args = parser.parse_args()
    print(f"Using model: {args.model_name}")
except SystemExit as e:
    print("Error parsing arguments. Please provide a valid --model_name argument.")
    raise

# Generate DeepSpeed configuration
def create_deepspeed_config():
    ds_config = {
        "train_micro_batch_size_per_gpu": "auto",  # Automatically determined
        "gradient_accumulation_steps": "auto",  # Automatically determined
        "fp16": {
            "enabled": "auto"  # Automatically determined
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": "auto",  # Automatically determined
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": "auto"  # Automatically determined
            }
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": "auto",  # Automatically determined
                "warmup_num_steps": "auto"  # Automatically determined
            }
        },
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {
                "device": "none"
            },
            "offload_param": {
                "device": "none"
            },
            "overlap_comm": True,
            "reduce_scatter": True,
            "contiguous_gradients": True
        }
    }
    with open("ds_config.json", "w") as f:
        json.dump(ds_config, f, indent=4)
    print("DeepSpeed config file created as 'ds_config.json'.")

# Create DeepSpeed config
create_deepspeed_config()

# Initialize distributed setup for DeepSpeed
init_distributed()

# Determine the current rank and assign the corresponding GPU
rank = torch.distributed.get_rank()
device = torch.device(f"cuda:{rank}")
torch.cuda.set_device(device)

# Load model and tokenizer
model_name = args.model_name  # Use model name from command-line argument
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

# Assign a padding token (use eos_token as pad_token)
tokenizer.pad_token = tokenizer.eos_token

# Load model and set pad_token_id
model = AutoModelForCausalLM.from_pretrained(model_name)
model.to(device)  # Explicitly move the model to the correct device
if model.config.pad_token_id is None:
    model.config.pad_token_id = tokenizer.pad_token_id

# Ensure sequence length compatibility
sequence_length = 4096

# Generate synthetic dataset
class RandomTextDataset(Dataset):
    def __init__(self, tokenizer, num_samples=100, seq_len=sequence_length):
        self.input_ids = torch.randint(
            0, tokenizer.vocab_size, (num_samples, seq_len), dtype=torch.long
        )

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {"input_ids": self.input_ids[idx], "labels": self.input_ids[idx]}


# Create dataset
num_samples = 1000
batch_size = 1
dataset = RandomTextDataset(tokenizer, num_samples=num_samples)

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
    deepspeed="./ds_config.json",
)

# Custom Trainer
class TokenSpeedTrainer(Trainer):
    def train(self, **kwargs):
        # Start timing
        start_time = time.time()
        super().train(**kwargs)  # Use Hugging Face's Trainer train method
        end_time = time.time()

        # Calculate throughput
        elapsed_time = end_time - start_time
        total_samples = len(self.train_dataset)  # Total samples in dataset
        sequence_length = 4096  # Match sequence length in the script
        batch_size = self.args.per_device_train_batch_size
        num_gpus = torch.cuda.device_count()  # Automatically detect GPUs

        # Total tokens processed
        total_tokens = total_samples * sequence_length * batch_size

        # Tokens per GPU per second
        # FIXME: this is a consequential update. I think these results are still single GPU
        # dividing by GPU is unnecessary

        # Original
        #tokens_per_second = total_tokens / (elapsed_time * num_gpus)

        # Updated
        tokens_per_second = total_tokens / elapsed_time
        print(f"Training tokens per GPU per second: {tokens_per_second:.2f}")
        return tokens_per_second


# Instantiate trainer
trainer = TokenSpeedTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=None  # Use the Trainer's default data collator
)

# Run training and calculate throughput
tokens_per_second = trainer.train()
print(f"Figure of merit: {tokens_per_second:.2f} tokens/second per GPU")
