from torch import Tensor, nn
from i2i.datasets.collator import I2IBatch


class ReconstructionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.reconstruction_loss = nn.L1Loss()

    def forward(self, batch: I2IBatch) -> Tensor:
        reconstruction_loss = self.reconstruction_loss(batch.target_images, batch.predicted_image)

        return reconstruction_loss
