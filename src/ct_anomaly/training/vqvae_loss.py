"""
Loss for the vector-quantized autoencoder.
"""

import torch


class VQVAELoss(torch.nn.Module):
    """
    Reconstruction loss plus the quantizer's codebook loss.
    """

    def __init__(self, reconstruction_loss_fn):
        super().__init__()
        self.reconstruction_loss_fn = reconstruction_loss_fn

        self.last_reconstruction_loss = 0.0
        self.last_kl_loss = 0.0
        self.last_perplexity = 0.0

    def forward(self, model_output, target):
        reconstruction, codebook_loss, _, perplexity, _ = model_output

        reconstruction_loss = self.reconstruction_loss_fn(reconstruction, target)

        self.last_reconstruction_loss = float(reconstruction_loss.detach())
        self.last_kl_loss = float(codebook_loss.detach())
        self.last_perplexity = float(perplexity.detach())

        return reconstruction_loss + codebook_loss
