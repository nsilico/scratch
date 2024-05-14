import torch
import torch.distributed as dist
import time
import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.nn.parallel import DistributedDataParallel as DDP

# Configuration dictionary
config = {
    "batch_size": 500,
    "num_batches": 64,
    "max_length": 50,  # Maximum length of generated text
    "sentence": "This is a test sentence. " * 50,
}

def setup(rank: int, world_size: int):
    print(f"[Rank {rank}] Setting up process group...")
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'  # CHANGED: Updated port to 29500
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    print(f"[Rank {rank}] Process group initialized.")

def cleanup(rank: int):
    print(f"[Rank {rank}] Cleaning up process group...")
    dist.destroy_process_group()
    print(f"[Rank {rank}] Process group cleaned up.")

def init_model(rank: int) -> GPT2LMHeadModel:
    """Initialize the GPT-2 model."""
    print(f"[Rank {rank}] Initializing GPT-2 model...")
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    print(f"[Rank {rank}] GPT-2 model initialized.")
    return model

def process_inputs(batch: List[str], model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer, device: torch.device, rank: int) -> Tuple[int, List[str]]:
    """Process a batch of inputs on a specific GPU."""
    print(f"[Rank {rank}] Tokenizing batch...")
    encoded_inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    print(f"[Rank {rank}] Tokenization complete. Running model...")

    total_output_tokens = 0
    generated_texts = []
    with torch.no_grad():
        for input_ids in encoded_inputs['input_ids']:
            output = model.generate(input_ids.unsqueeze(0), max_length=config["max_length"], num_return_sequences=1)
            output_text = tokenizer.decode(output[0], skip_special_tokens=True)
            generated_texts.append(output_text)
            total_output_tokens += len(output[0])

    print(f"[Rank {rank}] Model run complete.")
    return (total_output_tokens, generated_texts)

def main():
    rank_env = os.getenv('RANK')
    world_size_env = os.getenv('WORLD_SIZE')
    
    if rank_env is None or world_size_env is None:
        raise ValueError("RANK and WORLD_SIZE environment variables must be set")
    
    rank = int(rank_env)
    world_size = int(world_size_env)
    
    setup(rank, world_size)
    
    device = torch.device(f'cuda:{rank}')
    print(f"[Rank {rank}] Using device: {device}")

    model = init_model(rank).to(device)
    model = DDP(model, device_ids=[rank])
    print(f"[Rank {rank}] Model wrapped with DDP.")

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    input_batches = [[config["sentence"]] * config["batch_size"] for _ in range(config["num_batches"])]
    split_batches = [input_batches[i::world_size] for i in range(world_size)]
    local_batches = split_batches[rank]
    
    print(f"[Rank {rank}] Starting processing {len(local_batches)} batches.")
    
    total_output_tokens = 0
    example_outputs = []
    start_time = time.time()

    try:
        for i, batch in enumerate(local_batches):
            print(f"[Rank {rank}] Processing batch {i+1}/{len(local_batches)}...")
            batch_start_time = time.time()
            output_tokens, generated_texts = process_inputs(batch, model, tokenizer, device, rank)
            batch_time = time.time() - batch_start_time
            print(f"[Rank {rank}] Batch {i+1}/{len(local_batches)} processed in {batch_time:.2f} seconds.")
            total_output_tokens += output_tokens
            example_outputs.extend(generated_texts)
            print(f"[Rank {rank}] Finished batch {i+1}/{len(local_batches)}. Total output tokens so far: {total_output_tokens}")

        total_time = time.time() - start_time
        print(f"[Rank {rank}] All batches processed in {total_time:.2f} seconds. Reducing total output tokens.")
        total_output_tokens_tensor = torch.tensor(total_output_tokens, device=device)
        dist.reduce(total_output_tokens_tensor, dst=0, op=dist.ReduceOp.SUM)
        print(f"[Rank {rank}] Reduction completed.")

        if rank == 0:
            throughput = total_output_tokens_tensor.item() / total_time
            print(f"Total time taken: {total_time:.2f} seconds")
            print(f"Throughput: {throughput:.2f} tokens per second")
            print(f"Total output tokens: {total_output_tokens_tensor.item()}")
            print(f"Example outputs: {example_outputs[:5]}")  # Display first 5 examples
    finally:
        cleanup(rank)

if __name__ == "__main__":
    main()
