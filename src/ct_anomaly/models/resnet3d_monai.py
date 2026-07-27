from monai.networks.nets import resnet18, resnet34, resnet50

RESNET_DEPTHS = {
    18: resnet18,
    34: resnet34,
    50: resnet50,
}


def resnet3d(depth=18, num_classes=2):
    if depth not in RESNET_DEPTHS:
        raise ValueError(f"depth not in the list 18, 34 or 50")

    resnet_fn = RESNET_DEPTHS[depth]

    model = resnet_fn(
        spatial_dims=3,
        n_input_channels=1,
        num_classes=num_classes,
        conv1_t_stride=2,
    )
    return model