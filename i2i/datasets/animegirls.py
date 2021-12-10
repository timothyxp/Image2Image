from torch.utils.data import Dataset
import os
from PIL import Image

from torchvision import transforms


class AnimeGirls(Dataset):
    def __init__(self, dir_name: str, split='train', img_transforms=None):
        self.dir_name = os.path.join(dir_name, split)
        self.split = split

        self.files = os.listdir(self.dir_name)

        self.transforms = img_transforms
        if self.transforms is None:
            self.transforms = transforms.ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_path = os.path.join(self.dir_name, self.files[index])

        img = self.transforms(Image.open(img_path))

        border = img.shape[-1] // 2

        sketch = img[:, :, border:]
        target = img[:, :, :border]

        return target, sketch
