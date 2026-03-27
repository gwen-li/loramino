import torch

class BatchedLoRA(torch.nn.Module):
    def __init__(self, linear_layer, num_adapters, rank, alpha):
        """Connects to the next available port.

        Args:
        linear_layer: A port value greater or equal to 1024.

        Returns:
        The new minimum port.
        """
        super().__init__()

        self.linear_layer = linear_layer
        self.num_adapters = num_adapters
        self.rank = rank
        self.alpha = alpha

        noise = torch.randn(num_adapters, rank, linear_layer.in_features) * 0.01
        zeros = torch.zeros(num_adapters, linear_layer.out_features, rank)

        # Gaussian noise
        self.A = torch.nn.Parameter(noise)
        # Zero
        self.B = torch.nn.Parameter(zeros)
    
    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        device = x.device

        x = x.reshape(batch_size * seq_len, hidden_dim)

        adapter_ids = torch.arange(batch_size, device=device) % self.num_adapters
        adapter_ids = adapter_ids.repeat_interleave(seq_len)

        # FIX
        ## Inefficient - Change to group
        A = self.A[adapter_ids]
        B = self.B[adapter_ids]

        W0x = self.linear_layer(x)

        # Ax
        Ax = torch.bmm(A, x.unsqueeze(-1)).squeeze(-1)
        # BAx
        BAx = torch.bmm(B, Ax.unsqueeze(-1)).squeeze(-1)
        
        out = W0x + (self.alpha / self.rank) * BAx
        out = out.reshape(batch_size, seq_len, -1)

        return out

