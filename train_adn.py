import torch
from torch import nn
from tensorfn import load_arg_config, load_wandb
from tensorfn import distributed as dist
from tqdm import tqdm
from diffusion import GaussianDiffusion
from config import DiffusionConfig
import torchvision
import time_step_discriminator
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

def train(conf, loader, model, discriminator, ema, diffusion, optimizer, optimizer_discriminator, scheduler, scheduler_discriminator, device, wandb):
    loader = sample_data(loader)

    pbar = range(conf.training.n_iter + 1)
    pbar = tqdm(pbar, dynamic_ncols=True)
    real_label = 1.
    fake_label = 0.
    bce_loss = nn.BCELoss()
    gloss_before_ema = 0
    dloss_before_ema = 0
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

        discriminator.zero_grad()
        # Format batch
        b_size = img.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        generator_sample, true_sample, conditional_sample = diffusion.samples_and_noise(model, img, time)
        # Forward pass real batch through D
        conditional_changed = torch.cat((conditional_sample, (time[:, None, None, None]+1)/conf.diffusion.beta_schedule["n_timestep"]*torch.ones((conditional_sample.shape[0], 1, conditional_sample.shape[2], conditional_sample.shape[3])).to(device)), dim = 1)
        true_changed = torch.cat((true_sample, (time[:, None, None, None])/conf.diffusion.beta_schedule["n_timestep"]*torch.ones((true_sample.shape[0], 1, true_sample.shape[2], true_sample.shape[3])).to(device)), dim = 1)
        generator_changed = torch.cat((generator_sample, (time[:, None, None, None])/conf.diffusion.beta_schedule["n_timestep"]*torch.ones((generator_sample.shape[0], 1, generator_sample.shape[2], generator_sample.shape[3])).to(device)), dim = 1)
        output = discriminator(conditional_changed, true_changed).view(-1)
        # Calculate loss on all-real batch
        errD_real = bce_loss(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = discriminator(conditional_changed, generator_changed.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = bce_loss(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        scheduler_discriminator.step()
        optimizer_discriminator.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        model.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = discriminator(conditional_changed, generator_changed).view(-1)
        # Calculate G's loss based on this output
        errG = bce_loss(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        #nn.utils.clip_grad_norm_(model.parameters(), 1)
        scheduler.step()
        optimizer.step()
        i == conf.training.scheduler.warmup:
            gloss_before_ema = errG
            dloss_before_ema = errD
        accumulate(ema, model, 0 if i < conf.training.scheduler.warmup else 0.9999)

        lr = optimizer.param_groups[0]["lr"]
        pbar.set_description(f"epoch: {epoch}; discriminator loss: {errD:.4f}; generator loss: {errG:.4f}; lr: {lr:.5f}; gbe: {gloss_before_ema:4f}; gde: {dloss_before_ema:4f}")

        if wandb is not None and i % conf.evaluate.log_every == 0:
            wandb.log({"epoch": epoch, "discriminator loss": errD, "generator loss": errG, "lr": lr}, step=i)

        if i % conf.evaluate.save_every == 0:
            model_module = model
            torch.save(
                {
                    "model": model_module.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "ema": ema.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler_discriminator": scheduler_discriminator.state_dict(),
                    "optimizer_discriminator": optimizer_discriminator.state_dict(),
                    "conf": conf,
                },
                f"checkpoint/diffusion_{str(i).zfill(6)}.pt",
            )

def main(conf):
    print(torch.cuda.is_available())
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

    discriminator = time_step_discriminator.ConditionalTimeStepDiscriminator_DualHead(conf.discriminator)
    discriminator = discriminator.to(device)
    optimizer_discriminator = conf.training.optimizer.make(discriminator.parameters())
    scheduler_discriminator = conf.training.scheduler.make(optimizer_discriminator)

    optimizer = conf.training.optimizer.make(model.parameters())
    scheduler = conf.training.scheduler.make(optimizer)

    if conf.ckpt is not None:
        ckpt = torch.load(conf.ckpt, map_location=lambda storage, loc: storage)
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        discriminator.load_state_dict(ckpt["discriminator"])

    betas = conf.diffusion.beta_schedule.make()
    diffusion = GaussianDiffusion(betas).to(device)
    train(conf, train_loader, model, discriminator, ema, diffusion, optimizer, optimizer_discriminator, scheduler, scheduler_discriminator, device, wandb)

if __name__ == "__main__":
    conf = load_arg_config(DiffusionConfig)

    dist.launch(
        main, conf.n_gpu, conf.n_machine, conf.machine_rank, conf.dist_url, args=(conf,)
    )
    #python train.py --n_gpu 1 --conf config/diffusion.conf
