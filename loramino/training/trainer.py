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



def setup_model(config_options: dict):
    """ Loads the base model and replaces linear layers with LoRA modules,
        freezes the base model parameters.

        Args:
            config_options (dict): Configuration options for the model.

        Returns:
            model (torch.nn.Module): The modified model with BatchedLoRA modules.
    """
    device = None
    if 'device' in config_options:
        device = torch.device(config_options['device'])
    else:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
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
    data = config_options['dataset']
    dataloader = DataLoader(OrcaMath(data, model.tokenizer),
                            batch_size=config_options['batch_size'],
                            shuffle=True)
    optimizer_dict = {
        'adam': torch.optim.Adam,
        'sgd': torch.optim.SGD,
        'adamw': torch.optim.AdamW
    }
    optimizer_class = optimizer_dict[config_options['optimizer']]
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = optimizer_class(trainable_parameters, **config_options['optimizer_params'])
    loss_fn_dict = {
        'mse': torch.nn.MSELoss(),
        'cross_entropy': torch.nn.CrossEntropyLoss(),
        'cosine_similarity': torch.nn.CosineSimilarity()
    }
    
    loss_fn = loss_fn_dict[config_options['loss_function']]
    if config_options['verbose']:
        print(f"Starting training for {config_options['num_epochs']} epochs...")
    for epoch in range(config_options['num_epochs']):
        for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
        if config_options['verbose']:
            print(f"Epoch {epoch+1}/{config_options['num_epochs']} completed. Loss: {loss.item()}")
