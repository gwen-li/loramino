from time import perf_counter

import torch
from loramino.models import loader as load_model
from loramino.adapters.baseline_lora import BaselineLoRA
from loramino.adapters.batched_lora import BatchedLoRA
from torch.utils.data import DataLoader
from loramino.data.orca_math import OrcaMath
from tqdm import tqdm


ADAPTER_REGISTRY = {
    "baseline": BaselineLoRA,
    "baseline_lora": BaselineLoRA,
    "batched": BatchedLoRA,
    "batched_lora": BatchedLoRA,
}


def resolve_device(config_options: dict) -> torch.device:
    if 'device' in config_options:
        return torch.device(config_options['device'])
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _normalize_adapter_type(config_options: dict) -> str:
    adapter_type = config_options.get("adapter_type", "batched_lora")
    if adapter_type not in ADAPTER_REGISTRY:
        valid_types = ", ".join(sorted(ADAPTER_REGISTRY))
        raise ValueError(f"Unknown adapter_type '{adapter_type}'. Expected one of: {valid_types}")
    return adapter_type


def _build_adapter(linear_layer: torch.nn.Linear,
                   adapter_type: str,
                   lora_config: dict,
                   device: torch.device):
    adapter_class = ADAPTER_REGISTRY[adapter_type]
    adapter_kwargs = dict(lora_config)
    adapter_kwargs["device"] = device

    if adapter_class is BaselineLoRA:
        adapter_kwargs.pop("num_adapters", None)

    return adapter_class(linear_layer, **adapter_kwargs)


def _replace_linear_layers(module: torch.nn.Module,
                           adapter_type: str,
                           lora_config: dict,
                           device: torch.device) -> int:
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.Linear):
            setattr(module, name, _build_adapter(child, adapter_type, lora_config, device))
            replaced += 1
            continue
        replaced += _replace_linear_layers(child, adapter_type, lora_config, device)
    return replaced


def build_dataloader(config_options: dict, tokenizer, shuffle: bool = True) -> DataLoader:
    data = config_options.get('dataset', '')
    max_length = config_options.get('max_length', 256)
    dataset = OrcaMath(data, tokenizer, max_length=max_length)
    return DataLoader(
        dataset,
        batch_size=config_options['batch_size'],
        shuffle=shuffle,
    )


def build_optimizer(model: torch.nn.Module, config_options: dict):
    optimizer_dict = {
        'adam': torch.optim.Adam,
        'sgd': torch.optim.SGD,
        'adamw': torch.optim.AdamW
    }
    optimizer_class = optimizer_dict[config_options['optimizer']]
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return optimizer_class(trainable_parameters, **config_options['optimizer_params'])


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) for key, value in batch.items()}


def run_training_step(model: torch.nn.Module, batch: dict, optimizer):
    optimizer.zero_grad()
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    return outputs, loss



def setup_model(config_options: dict):
    """ Loads the base model and replaces linear layers with LoRA modules,
        freezes the base model parameters.

        Args:
            config_options (dict): Configuration options for the model.

        Returns:
            model (torch.nn.Module): The modified model with BatchedLoRA modules.
    """
    device = resolve_device(config_options)
    model = load_model.load_model(config_options['base_model'])
    lora_config = config_options['lora_config']
    adapter_type = _normalize_adapter_type(config_options)

    if isinstance(lora_config['alpha'], float):
        lora_config['alpha'] = torch.tensor(lora_config['alpha'])
    model.requires_grad_(False)
    num_lora_layers = _replace_linear_layers(model, adapter_type, lora_config, device)
    model.to(device)
    if config_options['verbose']:
        print(f"Replaced {num_lora_layers} linear layers with {adapter_type} modules.")

    return model


def train(config_options):

    # Load base model
    model = setup_model(config_options)
    device = next(model.parameters()).device
    dataloader = build_dataloader(config_options, model.tokenizer, shuffle=True)
    optimizer = build_optimizer(model, config_options)
    step_times = []
    if config_options['verbose']:
        print(f"Starting training for {config_options['num_epochs']} epochs...")
    train_start = perf_counter()
    for epoch in range(config_options['num_epochs']):
        for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
            batch = move_batch_to_device(batch, device)
            synchronize_device(device)
            step_start = perf_counter()
            outputs, loss = run_training_step(model, batch, optimizer)
            synchronize_device(device)
            step_times.append(perf_counter() - step_start)
        if config_options['verbose']:
            print(f"Epoch {epoch+1}/{config_options['num_epochs']} completed. Loss: {loss.item()}")
    total_time = perf_counter() - train_start
    metrics = {
        "final_loss": loss.item(),
        "num_steps": len(step_times),
        "total_time_s": total_time,
        "avg_step_time_s": (sum(step_times) / len(step_times)) if step_times else 0.0,
    }
    if config_options['verbose']:
        print(f"Training time: {metrics['total_time_s']:.4f}s total, {metrics['avg_step_time_s']:.4f}s/step")
    return metrics
