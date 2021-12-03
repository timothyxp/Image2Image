from tqdm.autonotebook import tqdm
import torch
from i2i.logger.wandb import WanDBWriter


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

        if i % config.get('grad_accum_steps', 1) == 0:
            optimizer.step()

        if i > config.get('len_epoch', 1e9):
            break

        if scheduler is not None:
            scheduler.step()


@torch.no_grad()
def evaluate(model, loader, config, vocoder, logger: WanDBWriter):
    model.eval()

    for batch in tqdm(iter(loader)):
        batch = batch.to(config['device'])

        batch = model(batch)

        for i in range(batch.melspec_prediction.shape[0]):
            logger.set_step(logger.step + 1, "val")

            reconstructed_wav = vocoder.inference(batch.melspec_prediction[i:i + 1].transpose(-1, -2)).cpu()

            logger.add_text("text", batch.transcript[i])
            logger.add_audio("audio", reconstructed_wav, sample_rate=22050)

