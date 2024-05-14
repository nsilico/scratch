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
    print(f"[Rank {rank}] Setting up process group...")
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    print(f"[Rank {rank}] Process group initialized.")

def cleanup():
    print("Cleaning up process group...")
    dist.destroy_process_group()
    print("Process group cleaned up.")

def init_model() -> BertModel:
    """Initialize the BERT model."""
    print("Initializing BERT model...")
    model = BertModel.from_pretrained('bert-base-uncased')
    print("BERT model initialized.")
    return model

def process_inputs(batch: List[str], model: nn.Module, device: torch.device) -> Tuple[int, List[float]]:
    """Proc
