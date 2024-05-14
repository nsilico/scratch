import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import BertTokenizer, BertModel
import time
import os
from typing import List, Tuple

# Configuration dictionary
config = {
    "batch_size": 500,
    "num_batches": 64,
    "sentence": "This is a test sentence. " * 50,
}

def setup(rank: int, world_size: int):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def init_model() -> BertModel:
    """Initialize the BERT model."""
    try:
        model = BertModel.from_pretrained('bert-base-uncased')
        return model
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        raise

def process_inputs(batch: List[str], model: nn.Module, device: torch.device) -> Tuple[int, List[float]]:
    """Process a batch of inputs on a specific GPU."""
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoded_inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    
    # Perform inference
    with torch.no_grad():
        outputs = model(**encoded_inputs)
    
    # Move the output to CPU before returning to avoid CUDA context issues
    output_tokens = outputs.last_hidden_state.size(0) * outputs.last_hidden_state.size(1)
    example_output = outputs.last_hidden_state[0][0][:5].cpu().numpy().tolist()
    
    return (output_tokens, example_output)

def main(rank: int, world_size: int):
    setup(rank, world_size)
    
    device = torch.device(f'cuda:{rank}')
    model = init_model().to(device)
    model = DDP(model, device_ids=[rank])

    input_batches = [[config["sentence"]] * config["batch_size"] for _ in range(config["num_batches"])]
    split_batches = [input_batches[i::world_size] for i in range(world_size)]
    local_batches = split_batches[rank]

    total_output_tokens = 0
    example_outputs = []
    start_time = time.time()

    for batch in local_batches:
        output_tokens, example_output = process_inputs(batch, model, device)
        total_output_tokens += output_tokens
        example_outputs.append(example_output)

    end_time = time.time()

    # Aggregate results from all processes
    total_output_tokens_tensor = torch.tensor(total_output_tokens, device=device)
    dist.reduce(total_output_tokens_tensor, dst=0, op=dist.ReduceOp.SUM)
    
    total_time = end_time - start_time

    if rank == 0:
        throughput = total_output_tokens_tensor.item() / total_time
        print(f"Total time taken: {total_time:.2f} seconds")
        print(f"Throughput: {throughput:.2f} tokens per second")
        print(f"Total output tokens: {total_output_tokens_tensor.item()}")
        print(f"Example outputs: {example_outputs}")

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
