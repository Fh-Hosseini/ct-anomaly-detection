"""
This module defines a 3D ResNet model for anomaly detection in CT scan, which serves as a baseline for our project.
The model is built using the MONAI library and supports different depths of ResNet (18, 34, or 50).
"""

from monai.networks.nets import resnet18, resnet34, resnet50

RESNET_DEPTHS = {
    18: resnet18,
    34: resnet34,
    50: resnet50,
}


def resnet3d(depth=18, num_classes=2):
    """
    Create a 3D ResNet model with the specified depth and number of output classes.

    Args:
        depth: Depth of the ResNet model. Supported values are 18, 34, and 50.
        num_classes: Number of output classes for the classification task.

    Returns:
        model: A 3D ResNet model instance.
    """

    if depth not in RESNET_DEPTHS:
        raise ValueError(
            f"Depth {depth} is not supported. Expected one of {sorted(RESNET_DEPTHS)}."
        )

    resnet_fn = RESNET_DEPTHS[depth]

    model = resnet_fn(
        spatial_dims=3,
        n_input_channels=1,
        num_classes=num_classes,
        conv1_t_stride=2,
    )
    return model
