import argparse
import json
from loramino.training.trainer import train


def validate_config(config):
    required_keys = ['base_model', 'batch_size', 'num_epochs', 'num_adaptors', 'learning_rate']
    for key in required_keys:
        if key not in config:
            raise ValueError(f'Missing required config option: {key}')
        if isinstance(config[key], (int | float)) and config[key] <= 0:
            raise ValueError(f'{key} must be greater than 0')
    if 'dataset' not in config and 'dataset_jobs' not in config and 'jobs' not in config:
        raise ValueError("Missing required config option: dataset, dataset_jobs, or jobs")


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
    validate_config(config_options)
    if args.verbose:
        print('Final training configuration:')
        print(json.dumps(config_options, indent=2))
    train(config_options)
