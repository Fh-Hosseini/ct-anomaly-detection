"""
3D Variational Autoencoder (VAE) model.
"""

import torch
import torch.nn as nn

from src.ct_anomaly.models.cae3d import Encoder3D, Decoder3D


class VariationalEncoder3D(Encoder3D):
    """
    3D convolutional encoder for a variational autoencoder (VAE).
    Outputs the mean and log-variance of the latent distribution.
    """

    def __init__(
        self,
        input_channels,
        encoder_channels,
        latent_channels,
        num_groups,
        negative_slope,
    ):

        super().__init__(
            input_channels=input_channels,
            encoder_channels=encoder_channels,
            latent_channels=latent_channels * 2,
            num_groups=num_groups,
            negative_slope=negative_slope,
        )
        self.latent_channels = latent_channels

    def forward(self, x):
        """
        Forward pass through the encoder.
        Returns the mean and log-variance of the latent distribution.
        """

        x = self.features(x)
        x = self.to_latent(x)

        mean = x[:, : self.latent_channels]
        log_variance = x[:, self.latent_channels :]
        log_variance = torch.clamp(log_variance, min=-10.0, max=10.0)

        return mean, log_variance


def reparameterize(mean, log_variance):
    """
    Reparameterization trick to sample from the latent distribution.
    """
    standard_deviation = torch.exp(0.5 * log_variance)
    noise = torch.randn_like(standard_deviation)
    return mean + standard_deviation * noise


class VAE3D(nn.Module):
    """
    3D Variational Autoencoder (VAE) model combining the encoder and decoder.
    """

    def __init__(
        self,
        input_channels,
        encoder_channels,
        latent_channels,
        num_groups,
        negative_slope,
        upsample_mode,
    ):
        super().__init__()

        self.encoder = VariationalEncoder3D(
            input_channels=input_channels,
            encoder_channels=encoder_channels,
            latent_channels=latent_channels,
            num_groups=num_groups,
            negative_slope=negative_slope,
        )

        self.decoder = Decoder3D(
            output_channels=input_channels,
            encoder_channels=encoder_channels,
            latent_channels=latent_channels,
            num_groups=num_groups,
            negative_slope=negative_slope,
            upsample_mode=upsample_mode,
        )

    def forward(self, x):
        """
        Forward pass through the VAE.
        Returns the reconstruction, mean, and log-variance of the latent distribution.
        """
        mean, log_variance = self.encoder(x)

        if self.training:
            latent = reparameterize(mean, log_variance)
        else:
            latent = mean

        reconstruction = self.decoder(latent)
        return reconstruction, mean, log_variance

    def encode(self, x):
        mean, _ = self.encoder(x)
        return mean
