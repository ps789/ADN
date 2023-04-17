import torch
from torch import nn
from tensorfn import load_arg_config, load_wandb
from tensorfn import distributed as dist
from tqdm import tqdm
from diffusion import GaussianDiffusion
from config import DiffusionConfig
import torchvision
import time_step_discriminator


# Sample data from dataloder
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

# Accumulate paramteres for EMA
def accumulate(model1, model2, decay=0.9999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)

# Main training loop
def train(conf, loader, generator, discriminator, ema, diffusion, optimizer_generator, optimizer_discriminator, 
          scheduler_generator, scheduler_discriminator, device, wandb):
    
    loader = sample_data(loader)
    pbar = range(conf.training.n_iter + 1)
    pbar = tqdm(pbar, dynamic_ncols=True)

    real_label = 1.
    fake_label = 0.
    bce_loss = nn.BCELoss()
    g_losses = []
    d_losses = []

    for i in pbar:
        epoch, img = next(loader)

        # For datasets which return features and labels: discard labels
        img = img[0]
        img = img.to(device)

        # Get random timesteps
        time = torch.randint(0, conf.diffusion.beta_schedule["n_timestep"], (img.shape[0],), device=device,)

        # Format batch
        b_size = img.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        x_t_gen, x_t, x_t_1 = diffusion.samples_and_noise(generator, img, time)

        # DEBUG: Show images
        # import numpy as np
        # import matplotlib.pyplot as plt
        # for i in range(len(x_t_1)):
        #     img_plot = np.transpose((x_t_1[i].cpu().detach().numpy() + 1) / 2, (1, 2, 0))
        #     plt.figure()
        #     plt.imshow(img_plot)
        #     plt.xlabel(time[i])
        # plt.show()
        # quit()

        # Forward pass real batch through D
        discriminator.zero_grad()
        x_t_1_changed = torch.cat((x_t_1, (time[:, None, None, None]+1)/conf.diffusion.beta_schedule["n_timestep"]*torch.ones((x_t_1.shape[0], 1, x_t_1.shape[2], x_t_1.shape[3])).to(device)), dim = 1)
        x_t_changed = torch.cat((x_t, (time[:, None, None, None])/conf.diffusion.beta_schedule["n_timestep"]*torch.ones((x_t.shape[0], 1, x_t.shape[2], x_t.shape[3])).to(device)), dim = 1)
        x_t_gen_changed = torch.cat((x_t_gen, (time[:, None, None, None])/conf.diffusion.beta_schedule["n_timestep"]*torch.ones((x_t_gen.shape[0], 1, x_t_gen.shape[2], x_t_gen.shape[3])).to(device)), dim = 1)
        output = discriminator(x_t_1, x_t).view(-1)

        # Calculate loss on all-real batch
        errD_real = bce_loss(output, label)

        # Calculate gradients for D in backward pass
        errD_real.backward()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        label.fill_(fake_label)

        # Classify all fake batch with D
        output = discriminator(x_t_1, x_t_gen.detach()).view(-1)

        # Calculate D's loss on the all-fake batch
        errD_fake = bce_loss(output, label)

        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()

        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake

        # Update D
        scheduler_discriminator.step()
        optimizer_discriminator.step()

        # Update G network: maximize log(D(G(z)))
        generator.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost

        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = discriminator(x_t_1, x_t_gen).view(-1)

        # Calculate G's loss based on this output
        errG = bce_loss(output, label)
        mse_error = torch.nn.functional.mse_loss(x_t_gen, x_t)
        errG.backward()

        # Update G
        nn.utils.clip_grad_norm_(generator.parameters(), 1)
        scheduler_generator.step()
        optimizer_generator.step()
        accumulate(ema, generator, 0 if i < conf.training.scheduler.warmup else 0.9999)

        # Save values
        lr = optimizer_generator.param_groups[0]["lr"]
        pbar.set_description(f"epoch: {epoch}; discriminator loss: {errD:.4f}; generator loss: {errG:.4f}; mse loss: {mse_error:.5f}; lr: {lr:0.5f}")

        if wandb is not None and i % conf.evaluate.log_every == 0:
            wandb.log({"epoch": epoch, "discriminator loss": errD, "generator loss": errG, "lr": lr}, step=i)

        g_losses.append(errG)
        d_losses.append(errD)

        if i % conf.evaluate.save_every == 0:
            torch.save(
                {
                    "generator": generator.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "ema": ema.state_dict(),

                    "scheduler_generator": scheduler_generator.state_dict(),
                    "optimizer_generator": optimizer_generator.state_dict(),
                    "scheduler_discriminator": scheduler_discriminator.state_dict(),
                    "optimizer_discriminator": optimizer_discriminator.state_dict(),
                    "conf": conf,

                    "disc_losses": d_losses,
                    "gen_losses": g_losses
                },
                f"checkpoint/cifar10/adn/adn_diffusion_{str(i).zfill(6)}.pt",
            )

def main(conf):
    wandb = None
    if conf.evaluate.wandb:
        wandb = load_wandb()
        wandb.init(project="denoising diffusion")

    device = "cuda"
    transforms = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    train_set = torchvision.datasets.CIFAR10(root='./cifar10', train = True, download = True,
                                             transform = transforms)
    train_sampler = dist.data_sampler(train_set, shuffle=True, distributed=conf.distributed)
    train_loader = conf.training.dataloader.make(train_set, sampler=train_sampler)

    generator = conf.generator.make()
    generator = generator.to(device)
    optimizer_generator = conf.training.optimizer.make(generator.parameters())
    scheduler_generator = conf.training.scheduler.make(optimizer_generator)

    ema = conf.generator.make()
    ema = ema.to(device)

    discriminator = time_step_discriminator.ConditionalTimeStepDiscriminator_DualHead(conf.discriminator)
    discriminator = discriminator.to(device)
    optimizer_discriminator = conf.training.optimizer.make(discriminator.parameters())
    scheduler_discriminator = conf.training.scheduler.make(optimizer_discriminator)

    if conf.ckpt is not None:
        ckpt = torch.load(conf.ckpt, map_location=lambda storage, loc: storage)

        scheduler_generator.load_state_dict(ckpt['scheduler_generator'])
        optimizer_generator.load_state_dict(ckpt['optimizer_generator'])
        scheduler_discriminator.load_state_dict(ckpt['scheduler_discriminator'])
        optimizer_discriminator.load_state_dict(ckpt['optimizer_discriminator'])

        generator.load_state_dict(ckpt["generator"])
        discriminator.load_state_dict(ckpt["discriminator"])
        ema.load_state_dict(ckpt["ema"])

    betas = conf.diffusion.beta_schedule.make()
    diffusion = GaussianDiffusion(betas).to(device)
    train(conf, train_loader, generator, discriminator, ema, diffusion, optimizer_generator, optimizer_discriminator, 
          scheduler_generator, scheduler_discriminator, device, wandb)

if __name__ == "__main__":
    conf = load_arg_config(DiffusionConfig)

    dist.launch(
        main, conf.n_gpu, conf.n_machine, conf.machine_rank, conf.dist_url, args=(conf,)
    )
    #python train_adn.py --n_gpu 1 --conf config/diffusion_adn.conf
