import os
import sys
from rddm_t2i import ResidualDiffusion, Unet, UnetRes,set_seed, Trainer


debug = False
if debug:
    save_every = 2
    sampling_timesteps = 10

    train_num_steps = 100
else:
    save_every = 1000
    sampling_timesteps = 10
    train_num_steps = 10000


train_folder = '/root/autodl-tmp/diffusers/examples/text_to_image/folder'
train_batch_size = 2
latent_image_channels = 4
latent_image_size = 32


model = UnetRes(
    dim=64,
    dim_mults=(1, 2, 4, 8),
    channels=latent_image_channels                      # 4 for latent space
)

diffusion = ResidualDiffusion(
    model,
    image_size=latent_image_size,
    timesteps=1000,
    sampling_timesteps=sampling_timesteps,
    loss_type='l2'
)

trainer = Trainer(
    diffusion,
    train_folder,
    train_batch_size=train_batch_size,
    num_samples=1,
    train_lr=2e-4,
    train_num_steps=train_num_steps,
    gradient_accumulate_every=2,
    save_every=save_every
)

trainer.train()
