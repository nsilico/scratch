def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = init_model().to(device)
    
    input_batches = [[config["sentence"]] * config["batch_size"] for _ in range(config["num_batches"])]
    
    total_output_tokens = 0
    example_outputs = []
    start_time = time.time()

    for batch in input_batches:
        output_tokens, example_output = process_inputs(batch, model, device)
        total_output_tokens += output_tokens
        example_outputs.append(example_output)

    total_time = time.time() - start_time
    throughput = total_output_tokens / total_time
    print(f"Total time taken: {total_time:.2f} seconds")
    print(f"Throughput: {throughput:.2f} tokens per second")
    print(f"Total output tokens: {total_output_tokens}")
    print(f"Example outputs: {example_outputs}")

if __name__ == "__main__":
    main()
