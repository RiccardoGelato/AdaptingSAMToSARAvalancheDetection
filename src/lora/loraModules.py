import torch
import torch.nn as nn


class LoRAConv2d(nn.Module):
    """
    Wraps an existing nn.Conv2d layer with a LoRA adapter. The adapter adds a low-rank update to 
    the frozen base convolution weights.

    lora_config must contain:
      - 'alpha': scaling factor.
      - 'rank': low-rank dimension.
    """
    def __init__(self, base_module: nn.Conv2d, r: int = 8, alpha: float = 1.0):
        super(LoRAConv2d, self).__init__()

        self.base_module = base_module

        # Freeze the original conv weights (we adapt via LoRA)
        for param in self.base_module.parameters():
            param.requires_grad = False

        # Get dimensions of the conv weight
        out_channels, in_channels, kH, kW = self.base_module.weight.shape

        # Register alpha and rank as buffers so they are on the right device/dtype
        self.alpha = alpha
        self.rank = r

        #initialize LoRA weights
        if r > 0:
            self.lora_A = nn.Parameter(
                torch.randn(
                    size=(out_channels, in_channels, kH, self.rank),
                    dtype=self.base_module.weight.dtype,
                    device=self.base_module.weight.device
                ) * 0.01
            )
            self.lora_B = nn.Parameter(
                torch.zeros(
                    size=(out_channels, in_channels, self.rank, kW),
                    dtype=self.base_module.weight.dtype,
                    device=self.base_module.weight.device
                )
            )
            #self.lora_A = nn.Parameter(
            #    self.base_module.weight.new_zeros((r * kH, in_channels * kW))
            #)
            #self.lora_B = nn.Parameter(
            #  self.base_module.weight.new_zeros((out_channels//self.base_module.groups*kW, r*kH))
            #)
            self.scaling = self.alpha / self.rank
            # Freezing the pre-trained weight matrix
            self.base_module.weight.requires_grad = False
       
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.rank > 0:
            return self.base_module._conv_forward(
                x, 
                self.base_module.weight + (self.lora_A @ self.lora_B).view(self.base_module.weight.shape) * self.scaling,
                self.base_module.bias
            )
        return self.base_module(x)


class LoRALinear(nn.Module):
    def __init__(self, base_module: nn.Linear, in_features, out_features, r: int = 8, alpha: float = 1.0, lora_dropout: float = 0.0):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.base_module = base_module

        if r > 0:
            # Initialize low-rank matrices
            self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.01)
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))
            self.scaling = alpha / r
            self.lora_dropout = nn.Dropout(p=lora_dropout)
            # Freeze original weights
            for param in self.base_module.parameters():
                param.requires_grad = False
        else:
            self.lora_A = None
            self.lora_B = None

    def forward(self, x):
        out = self.base_module(x)
        if self.r > 0:
            lora_out = (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            out = out + lora_out
        return out
    
def replace_linear_with_lora(module, target_layer_names, r=4, alpha=1.0):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear) and any(t in name for t in target_layer_names):
                setattr(module, name, LoRALinear(child, child.in_features, child.out_features, r=r, alpha=alpha))
            else:
                replace_linear_with_lora(child, target_layer_names, r, alpha)

def replace_conv2d_with_lora(module, target_layer_names, r=4, alpha=1.0):
        for name, child in module.named_children():
            if isinstance(child, nn.Conv2d) and any(t in name for t in target_layer_names):
                setattr(module, name, LoRAConv2d(child, r=r, alpha=alpha))
                print(f"Replaced {name} with LoRAConv2d")
            else:
                replace_conv2d_with_lora(child, target_layer_names, r, alpha)