import numpy as np
import torch.utils.data as udata

from torch import from_numpy
from typing import Any, Dict

from dataloaders.utils import is_coordinate_in_padded_area


def create_torch_loader(img: np.array,
                        gt: np.array,
                        hyperparams: Dict,
                        shuffle: Any = False):
    hyperparams.setdefault('is_3d', True)
    dataset = TorchDataLoader(img, gt, **hyperparams)
    return udata.DataLoader(dataset, batch_size=hyperparams["batch_size"], shuffle=shuffle)
# ----------------------------------------------------------------------------------------------------------------------


class TorchDataLoader(udata.Dataset):
    """ Generic class for a hyperspectral scene """

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
        super(TorchDataLoader, self).__init__()
        self.data = data
        self.label = gt
        self.name = hyperparams["net_name"]
        self.patch_size = hyperparams["patch_size"]
        self.ignored_labels = set(hyperparams["ignored_labels"])
        self.flip_augmentation = hyperparams["flip_augmentation"]
        self.radiation_augmentation = hyperparams["radiation_augmentation"]
        self.mixture_augmentation = hyperparams["mixture_augmentation"]
        self.center_pixel = hyperparams["center_pixel"]
        self.is_3d = hyperparams["is_3d"]

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
                if is_coordinate_in_padded_area(coordinates=(x, y), image_size=data.shape, padding_size=p)
            ]
        )
        self.labels = [self.label[x, y] for x, y in self.indices]
        np.random.shuffle(self.indices)

    @staticmethod
    def flip(*arrays):
        horizontal = np.random.random() > 0.5
        vertical = np.random.random() > 0.5
        if horizontal:
            arrays = [np.fliplr(arr) for arr in arrays]
        if vertical:
            arrays = [np.flipud(arr) for arr in arrays]
        return arrays

    @staticmethod
    def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1 / 25):
        alpha = np.random.uniform(*alpha_range)
        noise = np.random.normal(loc=0.0, scale=1.0, size=data.shape)
        return alpha * data + beta * noise

    def mixture_noise(self, data, label, beta=1 / 25):
        alpha1, alpha2 = np.random.uniform(0.01, 1.0, size=2)
        noise = np.random.normal(loc=0.0, scale=1.0, size=data.shape)
        data2 = np.zeros_like(data)
        for idx, value in np.ndenumerate(label):
            if value not in self.ignored_labels:
                l_indices = np.nonzero(self.labels == value)[0]
                l_indice = np.random.choice(l_indices)
                assert self.labels[l_indice] == value
                x, y = self.indices[l_indice]
                data2[idx] = self.data[x, y]
        return (alpha1 * data + alpha2 * data2) / (alpha1 + alpha2) + beta * noise

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
            data, label = self.flip(data, label)
        if self.radiation_augmentation and np.random.random() < 0.1:
            data = self.radiation_noise(data)
        if self.mixture_augmentation and np.random.random() < 0.2:
            data = self.mixture_noise(data, label)

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
        if self.patch_size > 1 and self.is_3d:
            # Make 4D data ((Batch x) Planes x Channels x Width x Height)
            data = data.unsqueeze(0)
        return data, label
