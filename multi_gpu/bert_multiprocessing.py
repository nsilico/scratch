import torch
from transformers import BertTokenizer, BertModel
import torch.multiprocessing as mp
import time
import logging
from typing import List, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration dictionary
config = {
    "batch_size": 500,
    "num_batches": 64,
    "sentence": "This is a test sentence. " * 50,
}

def init_model(gpu_id: int) -> BertModel:
    """Initialize the BERT model on a specific GPU."""
    try:
        model = BertModel.from_pretrained('bert-base-uncased').to(f"cuda:{gpu_id}").eval()
        return model
    except Exception as e:
        logger.error(f"Failed to initialize model on GPU {gpu_id}: {e}")
        raise

def process_inputs(batch: List[str], gpu_id: int, model: BertModel) -> Tuple[int, List[float]]:
    """Process a batch of inputs on a specific GPU."""
    device = f'cuda:{gpu_id}'
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoded_inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    
    # Perform inference
    with torch.no_grad():
        outputs = model(**encoded_inputs)
    
    # Move the output to CPU before returning to avoid CUDA context issues
    output_tokens = outputs.last_hidden_state.size(0) * outputs.last_hidden_state.size(1)
    example_output = outputs.last_hidden_state[0][0][:5].cpu().numpy().tolist()
    
    # Clear CUDA memory
    del encoded_inputs
    del outputs
    torch.cuda.empty_cache()
    
    return (output_tokens, example_output)

def gpu_worker(gpu_id: int, input_batches: List[List[str]], result_list: mp.Manager().list):
    """Worker function to process batches on a given GPU."""
    try:
        model = init_model(gpu_id)
        total_output_tokens = 0
        example_outputs = []
        
        for batch in input_batches:
            output_tokens, example_output = process_inputs(batch, gpu_id, model)
            total_output_tokens += output_tokens
            example_outputs.append(example_output)
        
        result_list.append((total_output_tokens, example_outputs))
    except Exception as e:
        logger.error(f"Error in GPU worker {gpu_id}: {e}")

def main():
    """Main function to execute multiprocessing workload."""
    input_batches = [[config["sentence"]] * config["batch_size"] for _ in range(config["num_batches"])]
    num_gpus = torch.cuda.device_count()
    
    split_batches = [input_batches[i::num_gpus] for i in range(num_gpus)]
    
    manager = mp.Manager()
    result_list = manager.list()
    
    processes = []
    start_time = time.time()
    
    for gpu_id in range(num_gpus):
        p = mp.Process(target=gpu_worker, args=(gpu_id, split_batches[gpu_id], result_list))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    end_time = time.time()
    
    total_output_tokens = sum(result[0] for result in result_list)
    example_outputs = [output for result in result_list for output in result[1]]
    
    total_time = end_time - start_time
    throughput = total_output_tokens / total_time
    
    logger.info(f"Total time taken: {total_time:.2f} seconds")
    logger.info(f"Throughput: {throughput:.2f} tokens per second")
    logger.info(f"Total output tokens: {total_output_tokens}")
    logger.info(f"Example outputs: {example_outputs}")

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()
