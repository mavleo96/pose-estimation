import numpy as np
import torchvision.transforms as transforms

import sonata

imagenet_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

sd_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

array_to_tensor_transform = transforms.Compose(
    [
        transforms.ToPILImage(),  # numpy HWC → PIL
        transforms.PILToTensor(),  # PIL → CHW uint8 [0,255]
    ]
)


def load_sonata_deterministic_transform(grid_size=0.02, coord_scale=0.01):
    config = [
        dict(type="CenterShift", apply_z=True),
        dict(
            type="GridSample",
            grid_size=grid_size,
            hash_type="fnv",
            mode="train",
            return_grid_coord=True,
            return_inverse=True,
        ),
        dict(type="NormalizeColor"),
        dict(type="ToTensor"),
        dict(
            type="Collect",
            keys=("coord", "grid_coord", "color", "inverse"),
            feat_keys=("coord", "color", "normal"),
        ),
    ]

    transform = sonata.transform.Compose(config)

    def wrapper(data_dict, seed=0):
        np.random.seed(seed)
        data_dict["coord"] *= coord_scale
        return transform(data_dict)

    return wrapper
