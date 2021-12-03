from torch import Tensor, nn
from i2i.datasets.collator import I2IBatch


class ReconstructionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_mse = nn.MSELoss()

    def forward(self, batch: I2IBatch) -> Tensor:
        reconstruction_loss = self.image_mse(batch.target_images, batch.predicted_image)

        return reconstruction_loss
