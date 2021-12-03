from torch.utils.data import Dataset
import pandas as  pd
import os
from PIL import Image

from torchvision import transforms


class FacadesDataset(Dataset):
    def __init__(self, dir_name: str, split='train'):
        self.dir_name = dir_name
        self.split = split

        meta = pd.read_csv(os.path.join(self.dir_name, 'metadata.csv'))
        meta = meta[meta.split == split]

        if split == 'train':
            meta['image_id'] = meta['image_id'].apply(lambda s: s[:-2])

        self.meta = meta \
            .sort_values(by=['domain']) \
            .groupby("image_id") \
            .agg({'image_path': tuple})

        self.transforms = transforms.ToTensor()

    def __len__(self):
        return self.meta.shape[0]

    def __getitem__(self, index):
        target_path, sketch_path = self.meta.iloc[index]['image_path']

        target_path = os.path.join(self.dir_name, target_path.replace('B', 'A'))
        sketch_path = os.path.join(self.dir_name, sketch_path.replace("A", 'B'))

        target_img = self.transforms(Image.open(target_path))
        sketch_img = self.transforms(Image.open(sketch_path))

        return target_img, sketch_img
