import os
import torch
import torch.distributed as dist

def setup(rank: int, world_size: int):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    print(f"[Rank {rank}] Process group initialized.")

def cleanup():
    dist.destroy_process_group()
    print(f"[Rank {rank}] Process group cleaned up.")

def main():
    rank_env = os.getenv('RANK')
    world_size_env = os.getenv('WORLD_SIZE')
    
    if rank_env is None or world_size_env is None:
        raise ValueError("RANK and WORLD_SIZE environment variables must be set")
    
    rank = int(rank_env)
    world_size = int(world_size_env)
    
    setup(rank, world_size)
    
    # Create a tensor and perform a collective operation
    tensor = torch.ones(1).cuda(rank)
    print(f"[Rank {rank}] Before all_reduce: {tensor}")
    
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"[Rank {rank}] After all_reduce: {tensor}")
    
    cleanup()

if __name__ == "__main__":
    main()
