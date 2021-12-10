import torch


class I2IBatch:
    def __init__(self, target_images, sketch_images):
        self.target_images = target_images
        self.sketch_images = sketch_images
        self.device = self.target_images.device
        self.predicted_image = None

    def to(self, device, non_blocking=True):
        self.device = device
        self.target_images = self.target_images.to(device, non_blocking=non_blocking)
        self.sketch_images = self.sketch_images.to(device, non_blocking=non_blocking)

        if self.predicted_image is not None:
            self.predicted_image = self.predicted_image.to(device, non_blocking=non_blocking)

        return self

    def pin_memory(self):
        self.target_images.pin_memory()
        self.sketch_images.pin_memory()


class Collator:
    def __init__(self):
        pass

    def __call__(self, images):
        target_images, sketch_images = list(zip(*images))

        target_images = torch.stack(target_images)
        sketch_images = torch.stack(sketch_images)

        return I2IBatch(target_images, sketch_images)
