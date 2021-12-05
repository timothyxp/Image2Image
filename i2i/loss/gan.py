from torch import Tensor, nn
import torch
from i2i.datasets.collator import I2IBatch
from typing import Tuple
from .reconstruction import ReconstructionLoss


class GANLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.reconstruction_loss = ReconstructionLoss()

        self.gan_loss = nn.BCEWithLogitsLoss()

    @staticmethod
    def get_gan_target(tensor, true_target):
        if true_target:
            return torch.ones(*tensor.shape)

        return torch.zeros(*tensor.shape)

    def forward(self, batch: I2IBatch, G, D, generator_step) -> Tuple[Tensor, Tensor]:
        if generator_step:
            fake = D(batch, False)

            loss_gan = self.gan_loss(fake, self.get_gan_target(fake, True))

            reconstruction = self.reconstruction_loss(batch)

            return loss_gan, reconstruction

        else:
            fake = D(batch, False)

            fake_loss = self.gan_loss(fake, self.get_gan_target(fake, False))

            true = D(batch, True)

            true_loss = self.gan_loss(true, self.get_gan_target(fake, True))

            return fake_loss / 2, true_loss / 2
