## DeepSpeed llama training

```bash
# Update and install necessary packages
apt update && apt install -y git wget libgl1 python3-pip

# Upgrade pip and install Python libraries
pip install --upgrade pip setuptools wheel

# Install PyTorch, Transformers, DeepSpeed, and other dependencies
pip install torch torchvision torchaudio \
    transformers \
    deepspeed \
    datasets \
    accelerate \
    tensorboard
```



```bash
cat <<EOT > ds_config.json
{
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 1,
    "zero_optimization": {
        "stage": 2
    },
    "fp16": {
        "enabled": true
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 1e-2
        }
    }
}
EOT
```





```python
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
from deepspeed import init_distributed
from torch.utils.data import DataLoader, Dataset
from transformers import TrainingArguments, Trainer


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

# Define data collator with explicit device placement
def collate_fn_with_device(batch, device):
    return {key: torch.tensor(value).to(device) for key, value in batch.items()}

# Wrap DataCollator to ensure compatibility with DeepSpeed
data_collator = lambda batch: collate_fn_with_device(batch, device)

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

# Custom Trainer to track tokens per second
class TokenSpeedTrainer(Trainer):
    def train(self, **kwargs):
        start_time = time.time()
        total_tokens = 0
        for step, batch in enumerate(self.get_train_dataloader()):
            # Debug tensor devices
            for key, value in batch.items():
                print(f"{key}: {value.device}")  # Debugging device mismatch
            
            # Move all tensors in the batch to the model's device
            batch = {key: value.to(self.model.device) for key, value in batch.items()}
            
            # Perform a training step
            outputs = self.training_step(self.model, batch)
            
            # Optimizer and scheduler step
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            
            # Count tokens
            total_tokens += batch["input_ids"].numel()
        end_time = time.time()
        elapsed_time = end_time - start_time
        tokens_per_second = total_tokens / elapsed_time
        print(f"Training tokens per GPU per second: {tokens_per_second}")
        return tokens_per_second


# Instantiate trainer
trainer = TokenSpeedTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Run training and calculate throughput
tokens_per_second = trainer.train()
print(f"Figure of merit: {tokens_per_second:.2f} tokens/second per GPU")
```

```bash
# Run
deepspeed --num_gpus=8 train_llama2.py
```



