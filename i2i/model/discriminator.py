from torch import nn
import torch
from i2i.datasets.collator import I2IBatch


class Discriminator(nn.Module):
    def __init__(self, input_ch_size: int, hidden_channels: int, max_hidden_channels: int, n_layers: int):
        super().__init__()
        layers = [
            nn.Conv2d(input_ch_size, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.GELU()
        ]

        cur_hidden = hidden_channels

        for i in range(1, n_layers):
            out_hidden = min(cur_hidden, max_hidden_channels)

            layers += [
                nn.Conv2d(cur_hidden, out_hidden, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_hidden),
                nn.GELU()
            ]

            cur_hidden = out_hidden

        layers += [
            nn.Conv2d(cur_hidden, cur_hidden, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(cur_hidden),
            nn.GELU(),
            nn.Conv2d(cur_hidden, 1, kernel_size=4, stride=1, padding=1)
        ]

        self.net = nn.Sequential(*layers)

    def forward(self, batch: I2IBatch, true: bool, detach: bool = False):
        x = torch.cat((
            batch.sketch_images,
            batch.target_images if true else batch.predicted_image
        ), dim=1)
        
        if detach:
            x = x.detach()

        output = self.net(x).mean(dim=-1).mean(dim=-1)

        return output
