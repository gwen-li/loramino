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
        self.linear_layer.requires_grad_(False)

        alpha_tensor = torch.as_tensor(alpha, dtype=linear_layer.weight.dtype, device=device)
        if alpha_tensor.numel() == 1:
            alpha_tensor = alpha_tensor.repeat(num_adapters)
        elif alpha_tensor.shape != (num_adapters,):
            raise ValueError(
                f"Alpha must be a scalar or a tensor of shape ({num_adapters},), got {tuple(alpha_tensor.shape)}"
            )
        self.register_buffer("alpha", alpha_tensor)

        parameter_kwargs = {
            "device": device,
            "dtype": linear_layer.weight.dtype,
        }

        # Gaussian noise
        self.A = torch.nn.Parameter(
            torch.randn(num_adapters, rank, linear_layer.in_features, **parameter_kwargs) * 0.01
        )
        # Zero
        self.B = torch.nn.Parameter(
            torch.zeros(num_adapters, linear_layer.out_features, rank, **parameter_kwargs)
        )

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

        scales = (self.alpha[adapter_ids] / self.rank).unsqueeze(-1)
        out = W0x + scales * BAx
        out = out.reshape(batch_size, seq_len, -1)

        return out
