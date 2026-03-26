import torch

class BatchedLoRA(torch.nn.Module):
    def __init__(self, linear_layer, num_adapters, rank, alpha):
        super().__init__()

        self.linear_layer = linear_layer
        self.num_adapters = num_adapters
        self.rank = rank
        self.alpha = alpha

        # Gaussian noise
        self.A = torch.nn.Parameter(torch.randn(num_adapters, rank, linear_layer.in_features) * 0.01)
        # Zero
        self.B = torch.nn.Parameter(torch.zeros(num_adapters, linear_layer.out_features, rank))
    
    def forward(self, x):
        
