"""
3D vector-quantized variational autoencoder.
"""

import torch
import torch.nn as nn

from src.ct_anomaly.models.cae3d import Encoder3D, Decoder3D


class VectorQuantizer(nn.Module):
    """
    Replace each latent position with its nearest codebook entry.

    num_codes: how many vectors the codebook holds
    code_dim: the dimension of each code vector, which must match the encoder's latent channels
    commitment_cost: weight of the loss term that encourages the encoder to commit to a code
    """

    def __init__(
        self,
        num_codes,
        code_dim,
        commitment_cost,
        revive_threshold=1.0,
        usage_decay=0.99,
        revive_protection=50.0,
    ):
        super().__init__()

        self.num_codes = num_codes
        self.code_dim = code_dim
        self.commitment_cost = commitment_cost

        self.codebook = nn.Embedding(num_codes, code_dim)

        # Initialize the codebook with a normal distribution.
        self.codebook.weight.data.normal_(mean=0.0, std=1.0)

        self.register_buffer("usage_count", torch.zeros(num_codes))
        self.revive_threshold = revive_threshold
        self.revive_protection = revive_protection
        self.usage_decay = usage_decay

    def forward(self, latent):
        """
        Quantize a latent tensor of shape (batch, code_dim, d, h, w).

        Returns:
            quantized: same shape as latent, carrying gradients to the encoder
            codebook_loss: term that trains the codebook and the encoder
            distances_to_nearest: per-position distance to the chosen code,
                shape (batch, 1, d, h, w), used as an anomaly signal
            perplexity: effective number of codes in use this batch
            indices: chosen code index per position, shape (batch, d, h, w)
        """
        with torch.amp.autocast(device_type="cuda", enabled=False):
            latent = latent.float()

            batch_size = latent.shape[0]
            spatial_shape = latent.shape[2:]

            # Flatten the spatial dimensions and move the channel dimension to the last position for distance computation.
            flat = latent.permute(0, 2, 3, 4, 1).contiguous()
            flat = flat.reshape(-1, self.code_dim)

            # Compute the squared Euclidean distance between each flattened latent vector and each codebook entry.
            codes = self.codebook.weight
            distances = (
                flat.pow(2).sum(dim=1, keepdim=True)
                - 2.0 * flat.matmul(codes.t())
                + codes.pow(2).sum(dim=1)
            )

            nearest_distance, nearest_index = distances.min(dim=1)
            nearest_distance = torch.clamp(nearest_distance, min=0.0)

            quantized_flat = self.codebook(nearest_index)

            # Compute the codebook loss, which consists of two terms: the codebook term and the commitment term. The codebook term encourages the codebook to be close to the encoder outputs, while the commitment term encourages the encoder outputs to commit to a specific code.
            codebook_term = torch.mean((quantized_flat - flat.detach()) ** 2)
            commitment_term = torch.mean((quantized_flat.detach() - flat) ** 2)
            codebook_loss = codebook_term + self.commitment_cost * commitment_term

            quantized = quantized_flat.reshape(
                batch_size,
                spatial_shape[0],
                spatial_shape[1],
                spatial_shape[2],
                self.code_dim,
            )
            quantized = quantized.permute(0, 4, 1, 2, 3).contiguous()

            # Use the straight-through estimator to allow gradients to flow through the quantization operation. The quantized tensor is used in the forward pass, but the gradients are computed as if the latent tensor was used, allowing the encoder to be trained effectively.
            quantized = latent + (quantized - latent).detach()

            distances_to_nearest = nearest_distance.reshape(
                batch_size, 1, spatial_shape[0], spatial_shape[1], spatial_shape[2]
            )

            # Compute the usage of each code in the codebook by creating a one-hot encoding of the nearest indices and averaging over the batch. The perplexity is calculated as the exponential of the negative sum of the usage multiplied by its logarithm, which gives an indication of how many codes are effectively being used in this batch.
            counts = torch.bincount(nearest_index, minlength=self.num_codes).float()
            usage = counts / nearest_index.numel()
            perplexity = torch.exp(-torch.sum(usage * torch.log(usage + 1e-10)))

            if self.training:
                self.usage_count.mul_(self.usage_decay).add_(counts)

                # Revive dead codes if their usage falls below the revive threshold. This helps to prevent the codebook from becoming underutilized and ensures that all codes have a chance to be used during training.
                dead = torch.nonzero(self.usage_count < self.revive_threshold).squeeze(
                    1
                )
                if dead.numel() > 0:
                    picks = torch.randint(
                        0, flat.shape[0], (dead.numel(),), device=flat.device
                    )
                    self.codebook.weight.data[dead] = flat[picks].detach()
                    self.usage_count[dead] = self.revive_protection

            indices = nearest_index.reshape(
                batch_size, spatial_shape[0], spatial_shape[1], spatial_shape[2]
            )

            return quantized, codebook_loss, distances_to_nearest, perplexity, indices


class VQVAE3D(nn.Module):
    """
    Vector-quantized autoencoder combining the CAE encoder and decoder with a discrete bottleneck.
    """

    def __init__(
        self,
        input_channels,
        encoder_channels,
        latent_channels,
        num_groups,
        negative_slope,
        upsample_mode,
        num_codes,
        commitment_cost,
    ):
        super().__init__()

        self.encoder = Encoder3D(
            input_channels=input_channels,
            encoder_channels=encoder_channels,
            latent_channels=latent_channels,
            num_groups=num_groups,
            negative_slope=negative_slope,
        )

        self.quantizer = VectorQuantizer(
            num_codes=num_codes,
            code_dim=latent_channels,
            commitment_cost=commitment_cost,
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
        Forward pass of the VQ-VAE.
        """
        latent = self.encoder(x)
        quantized, codebook_loss, distances, perplexity, indices = self.quantizer(
            latent
        )
        reconstruction = self.decoder(quantized)

        return reconstruction, codebook_loss, distances, perplexity, indices

    def encode(self, x):
        """
        Return the quantized latent, matching CAE3D.encode.
        """
        was_training = self.quantizer.training
        self.quantizer.eval()
        latent = self.encoder(x)
        quantized, _, _, _, _ = self.quantizer(latent)
        if was_training:
            self.quantizer.train()
        return quantized
