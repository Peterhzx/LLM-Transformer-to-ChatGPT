import argparse
import sys

from src.utils import NLPModelPipeline

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-C", "--config", help="Passing in config path")

    args = parser.parse_args()
    if not args.config:
        print("Missing -C CONFIG, --config CONFIG")
        sys.exit(-1)

    pipeline = NLPModelPipeline(args.config)
    pipeline.run_pipline()


"""
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