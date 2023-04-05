import torch
from torch import nn
from tensorfn import load_arg_config, load_wandb
from tensorfn import distributed as dist
from tqdm import tqdm
from diffusion import GaussianDiffusion
from config import DiffusionConfig
import torchvision

def sample_data(loader):
    loader_iter = iter(loader)
    epoch = 0

    while True:
        try:
            yield epoch, next(loader_iter)

        except StopIteration:
            epoch += 1
            loader_iter = iter(loader)

            yield epoch, next(loader_iter)

def accumulate(model1, model2, decay=0.9999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)

def train(conf, loader, model, ema, diffusion, optimizer, scheduler, device, wandb):
    loader = sample_data(loader)

    pbar = range(conf.training.n_iter + 1)
    pbar = tqdm(pbar, dynamic_ncols=True)

    for i in pbar:
        epoch, img = next(loader)

        # For datasets which return features and labels: discard labels
        img = img[0]
        img = img.to(device)

        time = torch.randint(
            0,
            conf.diffusion.beta_schedule["n_timestep"],
            (img.shape[0],),
            device=device,
        )

        loss = diffusion.p_loss(model, img, time)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        scheduler.step()
        optimizer.step()

        accumulate(ema, model, 0 if i < conf.training.scheduler.warmup else 0.9999)

        lr = optimizer.param_groups[0]["lr"]
        pbar.set_description(f"epoch: {epoch}; loss: {loss.item():.4f}; lr: {lr:.5f}")

        if wandb is not None and i % conf.evaluate.log_every == 0:
            wandb.log({"epoch": epoch, "loss": loss.item(), "lr": lr}, step=i)

        if i % conf.evaluate.save_every == 0:
            model_module = model
            torch.save(
                {
                    "model": model_module.state_dict(),
                    "ema": ema.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "conf": conf,
                },
                f"checkpoint/diffusion_{str(i).zfill(6)}.pt",
            )

def main(conf):
    wandb = None
    if conf.evaluate.wandb:
        wandb = load_wandb()
        wandb.init(project="denoising diffusion")

    device = "cuda"

    train_set = torchvision.datasets.CIFAR10(root='./cifar10', train = True, download = True,
                                             transform = torchvision.transforms.ToTensor())
    train_sampler = dist.data_sampler(train_set, shuffle=True, distributed=conf.distributed)
    train_loader = conf.training.dataloader.make(train_set, sampler=train_sampler)

    model = conf.model.make()
    model = model.to(device)
    ema = conf.model.make()
    ema = ema.to(device)

    optimizer = conf.training.optimizer.make(model.parameters())
    scheduler = conf.training.scheduler.make(optimizer)

    if conf.ckpt is not None:
        ckpt = torch.load(conf.ckpt, map_location=lambda storage, loc: storage)
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])

    betas = conf.diffusion.beta_schedule.make()
    diffusion = GaussianDiffusion(betas).to(device)

    train(conf, train_loader, model, ema, diffusion, optimizer, scheduler, device, wandb)

if __name__ == "__main__":
    conf = load_arg_config(DiffusionConfig)

    dist.launch(
        main, conf.n_gpu, conf.n_machine, conf.machine_rank, conf.dist_url, args=(conf,)
    )
    #python train.py --n_gpu 1 --conf config/diffusion.conf
