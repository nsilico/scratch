# ... (all imports and argument parsing)

# Initialize distributed setup for DeepSpeed
init_distributed()

# Determine rank and world size
rank = torch.distributed.get_rank()
world_size = torch.distributed.get_world_size()
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
dataset = RandomTextDataset(tokenizer, num_samples=num_samples, seq_len=sequence_length)
sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)

# DataLoader with DistributedSampler
data_loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    sampler=sampler,
    collate_fn=lambda batch: {
        key: torch.stack([example[key] for example in batch]).to(device, non_blocking=True)
        for key in batch[0]
    },
    num_workers=4,
    pin_memory=True
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

            # Calculate global throughput (average per GPU)
            global_tokens_per_gpu_per_second = total_tokens_processed / (world_size * elapsed_time)

            if rank == 0:
                print(f"[LOG] Average Tokens/GPU/Second: {global_tokens_per_gpu_per_second:.2f}")

            return local_tokens_per_second, elapsed_time, global_tokens_per_gpu_per_second

        except Exception as e:
            print(f"[RANK {rank}] Training failed with exception: {e}")
            raise

# Trainer instantiation
trainer = TokenSpeedTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=lambda batch: {
        key: torch.stack([example[key] for example in batch]).to(device, non_blocking=True)
        for key in batch[0]
    }
)

# Run Training
if rank == 0:
    print(f"[LOG] Test started at: {datetime.now()}")

tokens_per_second, duration, tokens_per_gpu_per_second = trainer.train()

# Final Summary (Rank 0 Only)
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
    print("="*40)
