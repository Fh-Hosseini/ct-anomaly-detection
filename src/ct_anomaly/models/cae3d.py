"""
3D Convolutional Autoencoder (CAE).
The encoder compresses the input data into a latent representation, while the decoder reconstructs the data from this representation.
"""

import torch.nn as nn


def make_conv_unit(in_channels, out_channels, stride, num_groups, negative_slope):
    """
    Creates a convolutional unit consisting of a 3D convolution, group normalization, and LeakyReLU activation.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Stride for the convolution.
        num_groups: Number of groups for group normalization.
        negative_slope: Negative slope for LeakyReLU activation.

    Returns:
        nn.Sequential: A sequential container of the convolutional unit.
    """
    conv = nn.Conv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )
    norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
    activation = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

    return nn.Sequential(conv, norm, activation)


class Encoder3D(nn.Module):
    """
    3D Encoder module for a convolutional autoencoder.

    Args:
        input_channels: Number of input channels.
        encoder_channels: List of output channels for each convolutional layer in the encoder.
        latent_channels: Number of channels in the latent representation.
        num_groups: Number of groups for group normalization.
        negative_slope: Negative slope for LeakyReLU activation.

    Attributes:
        features: Sequential container of convolutional layers for feature extraction.
        to_latent: Convolutional layer to reduce the number of channels to the latent representation
    """

    def __init__(
        self,
        input_channels,
        encoder_channels,
        latent_channels,
        num_groups,
        negative_slope,
    ):
        super().__init__()

        layers = []
        # Add the first convolutional layer with stride 1 to preserve spatial dimensions.
        layers.append(
            make_conv_unit(
                in_channels=input_channels,
                out_channels=encoder_channels[0],
                stride=1,
                num_groups=num_groups,
                negative_slope=negative_slope,
            )
        )

        # Add additional convolutional layers with stride 2 for downsampling.
        for i in range(1, len(encoder_channels)):
            layers.append(
                make_conv_unit(
                    in_channels=encoder_channels[i - 1],
                    out_channels=encoder_channels[i],
                    stride=2,
                    num_groups=num_groups,
                    negative_slope=negative_slope,
                )
            )

        # Create a sequential container for the convolutional layers.
        self.features = nn.Sequential(*layers)

        # Add a convolutional layer to reduce the number of channels to the latent representation.
        self.to_latent = nn.Conv3d(
            in_channels=encoder_channels[-1],
            out_channels=latent_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x):
        """
        Forward pass of the encoder.

        Args:
            x: Input tensor of shape (batch_size, input_channels, depth, height, width).
        Returns:
            Tensor of shape (batch_size, latent_channels, depth', height', width') representing the latent representation.
        """
        x = self.features(x)
        x = self.to_latent(x)
        return x


class Decoder3D(nn.Module):
    """
    3D Decoder module for a convolutional autoencoder.

    Args:
        output_channels: Number of output channels (matched with input_channels of the encoder).
        encoder_channels: List of output channels for each convolutional layer in the encoder (used to determine decoder architecture).
        latent_channels: Number of channels in the latent representation.
        num_groups: Number of groups for group normalization.
        negative_slope: Negative slope for LeakyReLU activation.
        upsample_mode: Mode for upsampling (e.g., 'nearest', 'trilinear').

    Attributes:
        from_latent: Convolutional layer to expand the latent representation to the first decoder layer
        features: Sequential container of convolutional layers for feature reconstruction.
        to_output: Convolutional layer to produce the final output with the desired number of channels.
    """

    def __init__(
        self,
        output_channels,
        encoder_channels,
        latent_channels,
        num_groups,
        negative_slope,
        upsample_mode,
    ):
        super().__init__()

        # Reverse the encoder channels to create the decoder architecture.
        decoder_channels = list(reversed(encoder_channels))

        # Add a convolutional layer to expand the latent representation to the first decoder layer.
        self.from_latent = nn.Conv3d(
            in_channels=latent_channels,
            out_channels=decoder_channels[0],
            kernel_size=1,
            stride=1,
            padding=0,
        )

        layers = []

        # Add convolutional layers with stride 1 and upsampling layers to reconstruct the spatial dimensions.
        for i in range(1, len(decoder_channels)):
            layers.append(
                make_conv_unit(
                    in_channels=decoder_channels[i - 1],
                    out_channels=decoder_channels[i],
                    stride=1,
                    num_groups=num_groups,
                    negative_slope=negative_slope,
                )
            )
            layers.append(
                nn.Upsample(
                    scale_factor=2,
                    mode=upsample_mode,
                    align_corners=False,
                )
            )

        self.features = nn.Sequential(*layers)

        # Add a convolutional layer to produce the final output with the desired number of channels.
        self.to_output = nn.Conv3d(
            in_channels=decoder_channels[-1],
            out_channels=output_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        """
        Forward pass of the decoder.

        Args:
            x: Input tensor of shape (batch_size, latent_channels, depth', height', width') representing the latent representation.
        Returns:
            Tensor of shape (batch_size, output_channels, depth, height, width) representing the reconstructed output.
        """

        x = self.from_latent(x)
        x = self.features(x)
        x = self.to_output(x)
        return x


class CAE3D(nn.Module):
    """
    3D Convolutional Autoencoder (CAE) module.

    Args:
        input_channels: Number of input channels.
        encoder_channels: List of output channels for each convolutional layer in the encoder.
        latent_channels: Number of channels in the latent representation.
        num_groups: Number of groups for group normalization.
        negative_slope: Negative slope for LeakyReLU activation.
        upsample_mode: Mode for upsampling (e.g., 'nearest', 'trilinear').

    Attributes:
        encoder: Instance of the Encoder3D class for encoding input data.
        decoder: Instance of the Decoder3D class for reconstructing data from the latent representation.
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

        self.encoder = Encoder3D(
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
        Forward pass of the CAE.

        Args:
            x: Input tensor of shape (batch_size, input_channels, depth, height, width).

        Returns:
            Tensor of shape (batch_size, input_channels, depth, height, width) representing the reconstructed output.
        """
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction

    def encode(self, x):
        """
        Encodes the input tensor into the latent representation.
        """
        return self.encoder(x)
