import time
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from deepspeed import init_distributed
from torch.utils.data import DataLoader, Dataset
from transformers import TrainingArguments, Trainer

# Generate DeepSpeed configuration
def create_deepspeed_config():
    ds_config = {
        "train_micro_batch_size_per_gpu": "auto",  # Let Hugging Face set this
        "gradient_accumulation_steps": "auto",  # Let Hugging Face set this
        "fp16": {
            "enabled": True
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 1e-5,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 1e-2
            }
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": 1e-5,
                "warmup_num_steps": 100
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
model_name = "meta-llama/Llama-2-7b-hf"
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
        # Start training using Hugging Face's Trainer implementation
        start_time = time.time()
        output = super().train(**kwargs)
        end_time = time.time()

        # Calculate throughput
        elapsed_time = end_time - start_time
        total_tokens = len(self.train_dataset) * self.args.per_device_train_batch_size
        tokens_per_second = total_tokens / elapsed_time
        print(f"Training tokens per GPU per second: {tokens_per_second}")
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
