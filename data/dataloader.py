import numpy as np
import torch.utils.data as udata

from abc import ABC
from torch import from_numpy
from typing import Any, Dict, Literal, Tuple


TYPE_MODELS = Literal['vanilla', 'KAN']
MODEL_NAMES = Literal['1DCNN', '3DCNN', 'NM3DCNN', 'SSFTT', 'DMC']


def get_dataloader(img: np.ndarray,
                   gt: np.ndarray,
                   model_name: MODEL_NAMES,
                   hyperparams: Dict,
                   shuffle: Any = False):

    models_dataloader = {
        '1DCNN': DataLoader1DCNN,
        '3DCNN': DataLoader3DCNN,
        'NM3DCNN': DataLoaderNM3DCNN,
        'SSFTT': DataLoaderSSFTT,
        'DMC': DataLoaderDMC
    }

    dataloader = models_dataloader[model_name]

    dataset = dataloader(img, gt, **hyperparams)

    return udata.DataLoader(dataset,
                            batch_size=hyperparams["batch_size"],
                            shuffle=shuffle)
# ----------------------------------------------------------------------------------------------------------------------


class ABCDataLoader(ABC):
    def __init__(self):
        raise NotImplemented()

    def __getitem__(self, item):
        raise NotImplemented()

    @staticmethod
    def is_coordinate_in_padded_area(coordinates: Tuple,
                                     image_size: Tuple,
                                     padding_size: int) -> bool:
        x, y = coordinates
        is_in_x = padding_size < x < image_size[0] - padding_size
        is_in_y = padding_size < y < image_size[1] - padding_size
        return is_in_x and is_in_y
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def flip(*arrays):
        horizontal = np.random.random() > 0.5
        vertical = np.random.random() > 0.5
        if horizontal:
            arrays = [np.fliplr(arr) for arr in arrays]
        if vertical:
            arrays = [np.flipud(arr) for arr in arrays]
        return arrays
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1 / 25):
        alpha = np.random.uniform(*alpha_range)
        noise = np.random.normal(loc=0.0, scale=1.0, size=data.shape)
        return alpha * data + beta * noise
    # ------------------------------------------------------------------------------------------------------------------


class DataLoader1DCNN(ABCDataLoader):

    def __init__(self,
                 data: np.ndarray,
                 gt: np.ndarray,
                 **hyperparams):
        """
        Args:
            data: 3D hyperspectral image
            gt: 2D array of labels
            patch_size: int, size of the spatial neighbourhood
            center_pixel: bool, set to True to consider only the label of the
                          center pixel
            data_augmentation: bool, set to True to perform random flips
            supervision: 'full' or 'semi' supervised algorithms
        """
        super(DataLoader1DCNN, self).__init__()
        self.data = data
        self.label = gt
        self.name = hyperparams["net_name"]
        self.patch_size = hyperparams["patch_size"]
        self.ignored_labels = set(hyperparams["ignored_labels"])
        self.flip_augmentation = hyperparams["flip_augmentation"]
        self.radiation_augmentation = hyperparams["radiation_augmentation"]
        self.center_pixel = hyperparams["center_pixel"]

        mask = np.ones_like(gt)
        for label in self.ignored_labels:
            mask[gt == label] = 0
        x_pos, y_pos = np.nonzero(mask)
        p = self.patch_size // 2
        # get all coordinates with padding of nonzeros labels
        self.indices = np.array(
            [
                (x, y)
                for x, y in zip(x_pos, y_pos)
                if ABCDataLoader.is_coordinate_in_padded_area(coordinates=(x, y), image_size=data.shape, padding_size=p)
            ]
        )
        self.labels = [self.label[x, y] for x, y in self.indices]
        np.random.shuffle(self.indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2  # left up bound
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size  # right down bound

        data = self.data[x1:x2, y1:y2]  # get patch
        label = self.label[x1:x2, y1:y2]

        if self.flip_augmentation and self.patch_size > 1:
            # Perform data augmentation (only on 2D patches)
            data, label = ABCDataLoader.flip(data, label)
        if self.radiation_augmentation and np.random.random() < 0.1:
            data = ABCDataLoader.radiation_noise(data)

        # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype="float32")
        label = np.asarray(np.copy(label), dtype="int64")

        # Load the data into PyTorch tensors
        data = from_numpy(data)
        label = from_numpy(label)
        # Extract the center label if needed
        if self.center_pixel and self.patch_size > 1:
            label = label[self.patch_size // 2, self.patch_size // 2]
        # Remove unused dimensions when we work with invidual spectrums
        elif self.patch_size == 1:
            data = data[:, 0, 0]
            label = label[0, 0]

        # Add a fourth dimension for 3D CNN
        if self.patch_size > 1:
            # Make 4D data ((Batch x) Planes x Channels x Width x Height)
            data = data.unsqueeze(0)
        return data, label


class DataLoader3DCNN(ABCDataLoader):
    pass


class DataLoaderNM3DCNN(ABCDataLoader):
    pass


class DataLoaderSSFTT(ABCDataLoader):
    pass


class DataLoaderDMC(ABCDataLoader):
    pass
