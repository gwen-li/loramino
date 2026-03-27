import argparse
import json

def main():
  parser = argparse.ArgumentParser(description='Fine-tune base model with multiple datasets (or multiple copies of same) using batched LoRA')
  parser.add_argument('--base_model', type=str, help='Base model to fine-tune', default='tinyllama')
  parser.add_argument('--dataset', type=str, help='Dataset to fine-tune on')
  parser.add_argument('--batch_size', type=int, help='Batch size for fine-tuning')
  parser.add_argument('--num_epochs', type=int, help='Number of epochs for fine-tuning')
  parser.add_argument('--num_adaptors', type=int, help='Number of adaptors to train')
  parser.add_argument('--config_file', type=str, help='Path to config file for training')
  parser.add_argument('--output_dir', type=str, help='Directory to save the adaptors')
  parser.add_argument('--verbose', action='store_true', help='Whether to print verbose logs during training')
  args = parser.parse_args()
  config_options = json.load(open(args.config_file))
  for key, value in vars(args).items():
    if key != 'base_model' and key != 'config_file' and value is not None:
      config_options[key] = value
  if args.verbose:
    print('Final training configuration:')
    print(json.dumps(config_options, indent=2))
  
  
  