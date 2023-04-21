import argparse

def get_args():

    parser = argparse.ArgumentParser(description='GAN Training')

    # Dataset/Dataloader parameters
    parser.add_argument('--dataset_root', type=str,
        default='../cifar10')
    parser.add_argument('--image_size', type=int,
        default=32)
    parser.add_argument('--batch_size', type=int,
        default=64)
    parser.add_argument('--num_workers', type=int,
        default=4)

    # GAN parameters
    parser.add_argument('--num_channels', type=int,
        default=3)
    parser.add_argument('--latent_size', type=int,
        default=512)
    parser.add_argument('--generator_features', type=int,
        default=256)
    parser.add_argument('--discriminator_features', type=int,
        default=64)
    parser.add_argument('--n_gan', type=int,
        default=2)

    # Training parameters
    parser.add_argument('--num_epochs', type=int,
        default=100)
    parser.add_argument('--lr', type=float,
        default=0.001)
    parser.add_argument('--beta1', type=float,
        default=0.5)
    parser.add_argument('--beta2', type=float,
        default=0.999)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--save_model_frequency', type=int,
        default=10)
    parser.add_argument('--checkpoint_path', type=str,
        default='checkpoint\\cifar10\\chain_gan')
    
    # Diffusion parameters
    parser.add_argument('--beta_schedule', type=str,
        default='custom')
    parser.add_argument('--src_diffusion_prob_start', type=float,
        default = 1)
    parser.add_argument('--src_diffusion_prob_end', type=float,
        default = 0.1)

    return parser.parse_args()