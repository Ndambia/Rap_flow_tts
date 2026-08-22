import torch.nn as nn


class Conv2dDiscriminator(nn.Module):
    """Mel-spectrogram-level Conv2d discriminator (paper §3.2, 'Adversarial Learning')."""

    def __init__(self, n_layers=4, base_channels=32):
        super().__init__()
        layers = []
        in_ch = 1
        ch = base_channels
        for i in range(n_layers):
            layers.append(nn.Conv2d(in_ch, ch, kernel_size=3, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.2))
            in_ch, ch = ch, min(ch * 2, 256)
        self.layers = nn.ModuleList(layers)
        self.out = nn.Conv2d(in_ch, 1, kernel_size=3, padding=1)

    def forward(self, mel):
        x = mel.unsqueeze(1)              # (B, 1, n_mels, T)
        feats = []
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                feats.append(x)
        out = self.out(x)
        return out, feats
