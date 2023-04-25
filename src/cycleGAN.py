import os 

import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import imageio

import src
from .diff_augment import DiffAugment
from .ssim_loss import SSIM


class CycleGAN:

    def __init__(self, cfg):

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        print(f'Using device: {self.device}')

        self.logger = SummaryWriter(os.path.join('logs', cfg.exp_name, 'runs'))
        self.out_dir = cfg.out_dir

        # set up model and configs        
        self.G_X2Y = getattr(src, cfg.G)().to(self.device)
        self.G_Y2X = getattr(src, cfg.G)().to(self.device)
        self.D_X = getattr(src, cfg.D)().to(self.device)
        self.D_Y = getattr(src, cfg.D)().to(self.device)
        
    def train(self, dataloader_X, dataloader_Y, cfg_train):

        g_params = list(self.G_X2Y.parameters()) + list(self.G_Y2X.parameters())
        d_params = list(self.D_X.parameters()) + list(self.D_Y.parameters())

        # Create optimizers for the generators and discriminators
        self.g_optimizer = optim.Adam(g_params, cfg_train.lr, [cfg_train.beta1, cfg_train.beta2])
        self.d_optimizer = optim.Adam(d_params, cfg_train.lr, [cfg_train.beta1, cfg_train.beta2])
        

        iter_X = iter(dataloader_X)
        iter_Y = iter(dataloader_Y)

        # Get some fixed data from domains X and Y for sampling.
        fixed_X = next(iter_X).to(self.device)
        fixed_Y = next(iter_Y).to(self.device)

        n_epochs = min(len(iter_X), len(iter_Y))

        for epoch in range(1, cfg_train.n_epochs + 1):

            if epoch % n_epochs == 0:
                iter_X = iter(dataloader_X)
                iter_Y = iter(dataloader_Y)

            images_X = next(iter_X)
            images_X = images_X.to(self.device)

            images_Y = next(iter_Y)
            images_Y = images_Y.to(self.device)

            # train discriminators
            D_X_loss = torch.mean((self.D_X(DiffAugment(images_X, cfg_train.policy)) - 1) ** 2)
            D_Y_loss = torch.mean((self.D_Y(DiffAugment(images_Y, cfg_train.policy)) - 1) ** 2)

            self.logger.add_scalar('D/XY/real', D_X_loss, epoch)
            self.logger.add_scalar('D/YX/real', D_Y_loss, epoch)
            d_real_loss = D_X_loss + D_Y_loss

            D_X_loss = torch.mean(self.D_Y(DiffAugment(self.G_X2Y(images_X), cfg_train.policy)) ** 2)
            D_Y_loss = torch.mean(self.D_X(DiffAugment(self.G_Y2X(images_Y), cfg_train.policy)) ** 2)
           
            self.logger.add_scalar('D/XY/fake', D_X_loss, epoch)
            self.logger.add_scalar('D/YX/fake', D_Y_loss, epoch)
            d_fake_loss = D_X_loss + D_Y_loss

            self.d_optimizer.zero_grad()
            d_total_loss = d_real_loss + d_fake_loss
            d_total_loss.backward()
            self.d_optimizer.step()
        

            # train generators
            gxy_loss = torch.mean((self.D_X(DiffAugment(self.G_Y2X(images_Y), cfg_train.policy)) - 1) ** 2)
            cycle_consistency_loss = torch.mean(torch.abs(images_Y - self.G_X2Y(self.G_Y2X(images_Y))))
            gxy_loss += cfg_train.lambda_cycle * cycle_consistency_loss
            self.logger.add_scalar('G/XY', gxy_loss, epoch)

          
            gyx_loss = torch.mean((self.D_Y(DiffAugment(self.G_X2Y(images_X), cfg_train.policy)) - 1) ** 2)
            cycle_consistency_loss = torch.mean(torch.abs(images_X - self.G_Y2X(self.G_X2Y(images_X))))
            gyx_loss += cfg_train.lambda_cycle * cycle_consistency_loss
            self.logger.add_scalar('G/YX', gyx_loss, epoch)

            # backprop the aggregated g losses and update G_XtoY and G_YtoX
            self.g_optimizer.zero_grad()
            g_loss = gxy_loss + gyx_loss
            g_loss.backward()
            self.g_optimizer.step()

            # Print the log info
            if epoch % cfg_train.log_step == 0:
                print(
                    'Iteration [{:5d}/{:5d}] | d_real_loss: {:6.4f} | '
                    'd_fake_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                        epoch, cfg_train.n_epochs, d_real_loss.item(),
                        d_fake_loss.item(), g_loss.item()
                    )
                )

            # Save the generated samples
            if epoch % cfg_train.sample_every == 0:
                self.save_samples(epoch, fixed_Y, fixed_X, os.path.join(self.out_dir, 'samples'))

            # Save the model parameters
            if epoch % cfg_train.checkpoint_every == 0:
                self.save_ckpt(epoch, os.path.join(self.out_dir, 'ckpt'))


    def save_ckpt(self, epoch, out_dir):
        
        os.makedirs(out_dir, exist_ok=True)
        torch.save(self.G_X2Y.state_dict(), os.path.join(out_dir, f'G_X2Y_iter{epoch}.pkl'))
        torch.save(self.G_Y2X.state_dict(), os.path.join(out_dir, f'G_Y2X_iter{epoch}.pkl'))
        torch.save(self.D_X.state_dict(), os.path.join(out_dir, f'D_X_iter{epoch}.pkl'))
        torch.save(self.D_Y.state_dict(), os.path.join(out_dir, f'D_Y_iter{epoch}.pkl'))


    def save_samples(self, epoch, fixed_Y, fixed_X, out_dir):
        """
        Saves samples from both generators X->Y and Y->X.
        """

        os.makedirs(out_dir, exist_ok=True)
        fake_X = self.G_Y2X(fixed_Y)
        fake_Y = self.G_X2Y(fixed_X)

        X, fake_X = fixed_X.cpu().detach().numpy(), fake_X.cpu().detach().numpy()
        Y, fake_Y = fixed_Y.cpu().detach().numpy(), fake_Y.cpu().detach().numpy()

        for i,(real,fake) in enumerate(zip(X, fake_X)):
            _, h, w = real.shape
            merged = np.concatenate((real, fake), axis=2).transpose(1, 2, 0)
            merged = np.uint8(255 * ( merged + 1) / 2)
            imageio.imwrite(os.path.join(out_dir, f'X-Y-{i}_ep{epoch}.png'), merged)

        for i,(real,fake) in enumerate(zip(Y, fake_Y)):
            _, h, w = real.shape
            merged = np.concatenate((real, fake), axis=2).transpose(1, 2, 0)
            merged = np.uint8(255 * ( merged + 1) / 2)
            imageio.imwrite(os.path.join(out_dir, f'Y-X-{i}_ep{epoch}.png'), merged)


    def evaluate(self,):
        ...