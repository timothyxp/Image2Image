from tqdm.autonotebook import tqdm
import torch
from i2i.logger.wandb import WanDBWriter
from i2i.datasets.collator import I2IBatch
from i2i.utils.utils import set_require_grad
from collections import defaultdict
import numpy as np


def log_images(batch: I2IBatch, logger: WanDBWriter):
    logger.add_image("sketch", batch.sketch_images[0].detach().cpu().permute(1, 2, 0).numpy())

    if batch.target_images is not None:
        logger.add_image("ground_true", batch.target_images[0].detach().cpu().permute(1, 2, 0).numpy())

    if batch.predicted_image is not None:
        logger.add_image("prediction", batch.predicted_image[0].detach().cpu().permute(1, 2, 0).numpy())


def train_epoch(model, optimizer, loader, loss_fn, config, scheduler=None, logger: WanDBWriter = None):
    model.train()

    for i, batch in enumerate(tqdm(iter(loader))):
        if logger is not None:
            logger.set_step(logger.step + 1, mode='train')

        batch = batch.to(config['device'], non_blocking=True)

        optimizer.zero_grad()

        batch = model(batch)

        loss = loss_fn(batch)

        loss.backward()
        optimizer.step()

        if logger is not None:
            l2_loss = loss.detach().cpu().numpy()

            logger.add_scalar("l2_loss", l2_loss)

        if i % config['log_train_step'] == 0 and logger is not None:
            log_images(batch, logger)

        if i % config.get('grad_accum_steps', 1) == 0:
            optimizer.step()

        if i > config.get('len_epoch', 1e9):
            break

        if scheduler is not None:
            scheduler.step()


def gan_train_epoch(
        G, D, optimizer_G, optimizer_D, loader, loss_fn,
        config, scheduler=None, logger: WanDBWriter = None
):
    G.train()
    D.train()

    for i, batch in enumerate(tqdm(iter(loader))):
        if logger is not None:
            logger.set_step(logger.step + 1, mode='train')

        batch = batch.to(config['device'], non_blocking=True)

        batch = G(batch)

        set_require_grad(D, True)
        set_require_grad(G, False)
        optimizer_D.zero_grad()

        fake_loss, true_loss = loss_fn(batch, G, D, generator_step=False)
        loss = fake_loss + true_loss
        loss.backward()
        optimizer_D.step()

        if logger is not None:
            fake_loss_np = fake_loss.detach().cpu().numpy()
            true_loss_np = true_loss.detach().cpu().numpy()
            total_loss_np = loss.detach().cpu().numpy()

            logger.add_scalar("D_fake_loss", fake_loss_np)
            logger.add_scalar("D_true_loss", true_loss_np)
            logger.add_scalar("D_total_loss", total_loss_np)

        set_require_grad(D, False)
        set_require_grad(G, True)
        optimizer_G.zero_grad()

        gan_loss, reconstruction = loss_fn(batch, G, D, generator_step=True)
        loss = gan_loss + reconstruction
        loss.backward()
        optimizer_G.step()

        if logger is not None:
            gan_loss_np = gan_loss.detach().cpu().numpy()
            reconstruction_np = reconstruction.detach().cpu().numpy()
            total_loss_np = loss.detach().cpu().numpy()

            logger.add_scalar("G_gan_loss", gan_loss_np)
            logger.add_scalar("G_reconstruction_loss", reconstruction_np)
            logger.add_scalar("G_total_loss", total_loss_np)

        if i % config['log_train_step'] == 0 and logger is not None:
            log_images(batch, logger)

        if i > config.get('len_epoch', 1e9):
            break

        if scheduler is not None:
            scheduler.step()


@torch.inference_mode()
def evaluate(model, loader, config, loss_fn, logger: WanDBWriter = None):
    model.eval()
    metrics = defaultdict(list)

    for i, batch in enumerate(tqdm(iter(loader))):
        if logger is not None:
            logger.set_step(logger.step + 1, mode='train')

        batch = batch.to(config['device'])

        batch = model(batch)

        loss = loss_fn(batch)

        metrics['loss'].append(loss.detach().cpu().numpy())

        if i % config['log_val_step'] == 0 and logger is not None:
            log_images(batch, logger)

    if logger is not None:
        for metric_name, metric_val in metrics.items():
            logger.add_scalar(metric_name, np.mean(metric_val))

    return metrics
