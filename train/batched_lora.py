import torch

class BatchedLoRA(torch.nn.Module):
    def __init__(self,
                linear_layer: torch.nn.Linear,
                num_adapters: int = 1,
                rank: int = 1,
                alpha: float | torch.Tensor = 1.0,
                device: torch.device = torch.device('cpu')):
        super().__init__()
        self.linear_layer = linear_layer
        self.num_adapters = num_adapters
        self.rank = rank

        self.alpha = None

        if isinstance(alpha, float):
            self.alpha = torch.full((num_adapters, ), alpha, device=device)
        else:
            assert_fail_str = f"Alpha must be a scalar or a tensor of shape ({num_adapters},), got {self.alpha.shape}"
            assert self.alpha.shape == (num_adapters,), assert_fail_str

        # Gaussian noise
        self.A = torch.nn.Parameter(torch.randn(num_adapters, rank, linear_layer.in_features) * 0.01)
        # Zero
        self.B = torch.nn.Parameter(torch.zeros(num_adapters, linear_layer.out_features, rank))
        self.to(device)
        
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

