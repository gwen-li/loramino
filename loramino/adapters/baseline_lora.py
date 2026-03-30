import torch


class BaselineLoRA(torch.nn.Module):
    def __init__(self,
                linear_layer: torch.nn.Linear,
                rank: int = 1,
                alpha: float | torch.Tensor = 1.0,
                device: torch.device = torch.device("cpu")):
        super().__init__()
        self.linear_layer = linear_layer
        self.rank = rank
        self.linear_layer.requires_grad_(False)

        alpha_tensor = torch.as_tensor(alpha, dtype=linear_layer.weight.dtype)
        if alpha_tensor.numel() != 1:
            raise ValueError("BaselineLoRA alpha must be a scalar.")
        self.register_buffer("alpha", alpha_tensor.reshape(()))

        parameter_kwargs = {
            "device": device,
            "dtype": linear_layer.weight.dtype,
        }
        # Match the current BatchedLoRA initialization.
        self.A = torch.nn.Parameter(
            torch.randn(rank, linear_layer.in_features, **parameter_kwargs) * 0.01
        )
        self.B = torch.nn.Parameter(
            torch.zeros(linear_layer.out_features, rank, **parameter_kwargs)
        )

    def forward(self, x):
        base_output = self.linear_layer(x)
        lora_input = x.to(dtype=self.A.dtype)
        Ax = torch.matmul(lora_input, self.A.transpose(-1, -2))
        BAx = torch.matmul(Ax, self.B.transpose(-1, -2))
        return base_output + (self.alpha / self.rank) * BAx
