import torch
import load_model
from batch import BatchedLoRA
from torch.utils.data import DataLoader




def train(config_options):
    # Load the base model
    model = load_model.load_model(config_options['base_model'])
    # Create the batched LoRA module
    # lora_module = BatchedLoRA(model, config_options['num_adaptors'], config_options['rank'], config_options['alpha'])
    # Create the dataloader
    # dataloader = DataLoader(config_options['dataset'], batch_size=config_options['batch_size'], shuffle=True)
    # Create the optimizer
    # optimizer = torch.optim.Adam(lora_module.parameters(), lr=config_options['learning_rate'])
    # Train the model
    # for epoch in range(config_options['num_epochs']):
    #     for batch in dataloader:
    #         optimizer.zero_grad()
    #         outputs = lora_module(batch)
    #         loss = compute_loss(outputs, batch)
    #         loss.backward()
    #         optimizer.step()