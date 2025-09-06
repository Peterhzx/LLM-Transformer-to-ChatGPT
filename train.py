import argparse
import sys

from src.utils import NLPModelPipeline

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Passing in config path")
    parser.add_argument("-m", "--mode", help="Currently have 2 modes: 'sagemaker' or 'local'")

    args = parser.parse_args()

    if args.mode not in ["sagemaker", "local"]:
        raise ValueError(f"Invalid mode: {args.mode}")

    pipeline = NLPModelPipeline(args.config, args.mode)
    pipeline.run_pipline()


"""
loss 4.7510

import argparse

# Initialize the parser
parser = argparse.ArgumentParser(description="Sample script with flags.")

# Add flags
parser.add_argument("-m", "--message", help="Display a custom message")
parser.add_argument("-H", "--host", help="Specify a host address")

# Parse arguments
args = parser.parse_args()

# Use the flags
if args.message:
    print(f"Message: {args.message}")
if args.host:
    print(f"Host: {args.host}")
    
    
    import sys
    if len(sys.argv) < 2:
        print("Usage: python train.py <config_path>")
        sys.exit(1)
config = load_config(sys.argv[1])
dataloader = load_data(config["Dataloader"])
tokenizer = train_tokenizer(dataloader, config["Tokenizer"])
# save_tokenizer(tokenizer)
tokenize_data(dataloader, tokenizer, config["Tokenizer"])
trainer, test_loader = train_and_save_model(dataloader, config["Trainer"])
eval_model(config["Evaluator"], test_loader, tokenizer, trainer)
"""