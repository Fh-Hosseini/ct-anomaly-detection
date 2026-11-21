"""
This module contains a function to build an autoencoder model based on the given parameters.
The function supports building different types of autoencoders, including convolutional autoencoders (CAE),
variational autoencoders (VAE), and vector quantized variational autoencoders (VQVAE).
"""

from src.ct_anomaly.models.cae3d import CAE3D
from src.ct_anomaly.models.vae3d import VAE3D
from src.ct_anomaly.models.vqvae3d import VQVAE3D


def build_ae_model(
    model_type,
    input_channels,
    encoder_channels,
    latent_channels,
    num_groups,
    negative_slope,
    upsample_mode,
    num_codes=None,
    commitment_cost=0.25,
):
    """
    Builds an autoencoder model based on the specified parameters.

    Args:
        model_type: Type of the model to build ('cae', 'vae', 'vqvae').
        input_channels: Number of input channels.
        encoder_channels: List of output channels for each convolutional layer in the encoder.
        latent_channels: Number of channels in the latent representation.
        num_groups: Number of groups for group normalization.
        negative_slope: Negative slope for LeakyReLU activation.
        upsample_mode: Mode for upsampling (e.g., 'nearest', 'trilinear').
        num_codes: Size of the codebook. Required when model_type is 'vqvae', ignored otherwise.
        commitment_cost: Commitment cost for VQVAE (default is 0.25).

    Returns:
        An instance of the specified model type.
    """

    # Define parameters that are shared across all AE model types
    shared_params = {
        "input_channels": input_channels,
        "encoder_channels": encoder_channels,
        "latent_channels": latent_channels,
        "num_groups": num_groups,
        "negative_slope": negative_slope,
        "upsample_mode": upsample_mode,
    }

    if model_type == "cae":
        return CAE3D(**shared_params)

    if model_type == "vae":
        return VAE3D(**shared_params)

    if model_type == "vqvae":
        if num_codes is None:
            raise ValueError("num_codes is required for model_type 'vqvae'")
        return VQVAE3D(
            num_codes=num_codes, commitment_cost=commitment_cost, **shared_params
        )

    raise ValueError(f"Unknown model type: {model_type}")
