import torch.nn as nn
from torch.nn.utils import weight_norm as apply_weight_norm
import torch


class Decoder(nn.Module):
    def __init__(
        self,
        dims,
        dropout=None,
        dropout_prob=0.1,
        norm_layers=(),
        latent_in=(),
        weight_norm=True,
        use_tanh=True
    ):
        super(Decoder, self).__init__()

        self.input_dim = 3
        self.layer_dims = [self.input_dim] + list(dims) + [1]
        self.num_layers = len(self.layer_dims) - 1
        self.latent_in = set(latent_in)
        self.dropout_layers = set(dropout) if dropout is not None else set()
        self.norm_layers = set(norm_layers)
        self.use_weight_norm = weight_norm

        self.layers = nn.ModuleList()
        for layer_idx in range(self.num_layers):
            in_dim = self.layer_dims[layer_idx]
            if layer_idx in self.latent_in:
                in_dim += self.input_dim
            out_dim = self.layer_dims[layer_idx + 1]
            linear = nn.Linear(in_dim, out_dim)
            if self.use_weight_norm and layer_idx in self.norm_layers:
                linear = apply_weight_norm(linear)
            self.layers.append(linear)

        self.dropout = nn.Dropout(dropout_prob)
        self.th = nn.Tanh() if use_tanh else nn.Identity()
        self.prelu = nn.PReLU()


    # input: N x 3
    def forward(self, input):
        x = input
        for layer_idx, layer in enumerate(self.layers):
            if layer_idx in self.latent_in:
                x = torch.cat([x, input], dim=1)

            x = layer(x)

            # Keep the last layer linear before the output tanh.
            if layer_idx < self.num_layers - 1:
                x = self.prelu(x)
                if layer_idx in self.dropout_layers:
                    x = self.dropout(x)

        return self.th(x)
