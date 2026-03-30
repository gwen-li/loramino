import torch
import load_model
from .batched_lora import BatchedLoRA
from torch.utils.data import DataLoader
from utils.orca_math import OrcaMath
from tqdm import tqdm




def setup_model(config_options: dict):
    """ Loads the base model and replaces linear layers with BatchedLoRA modules,
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
    
    if (isinstance(lora_config['alpha'], float)):
        lora_config['alpha'] = torch.tensor(lora_config['alpha'])
    num_lora_layers = 0
    for name, module in model.named_modules():
        module.requires_grad_(False)
        if isinstance(module, torch.nn.Linear):
            num_lora_layers += 1
            setattr(model, name, BatchedLoRA(module, **{**lora_config, 'device': device})) 
    if config_options['verbose']:
        print(f"Replaced {num_lora_layers} linear layers with BatchedLoRA modules.")
    return model

def train(config_options):
    
    # Load base model
    model = setup_model(config_options)
    data = config_options['dataset']
    dataloader = DataLoader(OrcaMath(data, model.tokenizer),
                            batch_size=config_options['batch_size'],
                            shuffle=True)
    optimizer_dict = {
        'adam': torch.optim.Adam,
        'sgd': torch.optim.SGD,
        'adamw': torch.optim.AdamW,
        'muon': torch.optim.Muon
    }
    optimizer_class = optimizer_dict[config_options['optimizer']]
    optimizer = optimizer_class(model.parameters(), **config_options['optimizer_params'])
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
            optimizer.zero_grad()
            outputs = model(batch)
            loss = loss_fn(outputs, batch, reduction='none')
            loss.backward()
            optimizer.step()
        if config_options['verbose']:
            print(f"Epoch {epoch+1}/{config_options['num_epochs']} completed. Loss: {loss.item()}")
    
    