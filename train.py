import sys


from src.utils import train_tokenizer, load_config, load_data, save_tokenizer

if __name__ == '__main__':  # .\\config\\config.json
    if len(sys.argv) < 2:
        print("Usage: python train.py <config_path>")
        sys.exit(1)
    config = load_config(sys.argv[1])
    dataloader = load_data(config["Dataloader"])
    tokenizer = train_tokenizer(dataloader, config["Tokenizer"])
    save_tokenizer(tokenizer)

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
"""