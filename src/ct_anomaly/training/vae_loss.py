"""Loss functions for the variational autoencoder."""

import torch


def kl_divergence(mean, log_variance, free_bits, num_input_voxels):
    """
    Compute the KL divergence between a Gaussian and a standard normal distribution.
    """

    mean = mean.float()
    log_variance = log_variance.float()
    per_element = -0.5 * (1.0 + log_variance - mean.pow(2) - log_variance.exp())

    per_channel = per_element.mean(dim=(0, 2, 3, 4))
    if free_bits <= 0.0:
        batch_size = mean.shape[0]
        per_sample = per_element.reshape(batch_size, -1).sum(dim=1)
        return per_sample.mean() / num_input_voxels, per_channel.detach()

    charged = torch.clamp(per_channel - free_bits, min=0.0)
    num_positions = per_element.shape[2] * per_element.shape[3] * per_element.shape[4]
    return charged.sum() * num_positions / num_input_voxels, per_channel.detach()


class VAELoss(torch.nn.Module):
    """
    Reconstruction loss plus beta-weighted KL, with free bits and annealing.
    """

    def __init__(
        self,
        reconstruction_loss_fn,
        beta,
        num_input_voxels,
        free_bits=0.0,
        anneal_epochs=0,
    ):
        super().__init__()
        self.reconstruction_loss_fn = reconstruction_loss_fn
        self.target_beta = beta
        self.anneal_epochs = anneal_epochs
        self.beta = 0.0 if anneal_epochs > 0 else beta
        self.num_input_voxels = num_input_voxels
        self.free_bits = free_bits
        self.last_reconstruction_loss = 0.0
        self.last_kl_loss = 0.0
        self.last_per_channel_kl = None

    def set_epoch(self, epoch):
        """
        Linearly ramp beta from 0 to the target over anneal_epochs.
        """
        if self.anneal_epochs <= 0:
            self.beta = self.target_beta
            return
        fraction = min(1.0, float(epoch) / float(self.anneal_epochs))
        self.beta = self.target_beta * fraction

    def forward(self, model_output, target):
        reconstruction, mean, log_variance = model_output
        reconstruction_loss = self.reconstruction_loss_fn(reconstruction, target)
        kl_loss, per_channel_kl = kl_divergence(
            mean, log_variance, self.free_bits, self.num_input_voxels
        )
        self.last_reconstruction_loss = float(reconstruction_loss.detach())
        self.last_kl_loss = float(kl_loss.detach())
        self.last_per_channel_kl = per_channel_kl.cpu()
        return reconstruction_loss + self.beta * kl_loss
