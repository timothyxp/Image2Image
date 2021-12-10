import shutil
import subprocess
import re
import os
from typing import List, Dict
from i2i.datasets.collator import I2IBatch
from i2i.utils.utils import convert_to_pil


class FidCalculator:
    name = 'fid'

    def __init__(self, swap_dir_name):
        self.swap_dir_name = swap_dir_name

    def calculate(self, batches: List[I2IBatch]) -> Dict[str, List[float]]:
        if os.path.exists(self.swap_dir_name):
            shutil.rmtree(self.swap_dir_name)

        os.mkdir(self.swap_dir_name)

        pred_dir = os.path.join(self.swap_dir_name, "prediction")
        os.mkdir(pred_dir)

        true_dir = os.path.join(self.swap_dir_name, "true")
        os.mkdir(true_dir)

        i = 0
        for batch in batches:
            for j in range(batch.predicted_image.shape[0]):
                pred_img = convert_to_pil(batch.predicted_image[j])
                true_img = convert_to_pil(batch.target_images[j])

                pred_img.save(os.path.join(pred_dir, f"{i}.png"))
                true_img.save(os.path.join(true_dir, f"{i}.png"))

                i += 1

        process = subprocess.Popen(f'python -m pytorch_fid {pred_dir} {true_dir} --device cpu', shell=True,
                                   stdout=subprocess.PIPE)
        out, err = process.communicate()

        fid = float(re.findall(r'(?<=FID: {2})\d+.\d+', out.decode('utf-8'))[-1])

        print(f"FID: {fid:.3f}")

        return {self.name: [fid]}
