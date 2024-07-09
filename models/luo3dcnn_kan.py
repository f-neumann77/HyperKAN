from typing import Any, Optional, Dict
import numpy as np

from models.kan_layers import FastKANConvLayer, KAN
from models.model import Model

import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


class Luo3DCNN_KAN_Net(nn.Module):
    """
    HSI-CNN: A Novel Convolution Neural Network for Hyperspectral Image
    Yanan Luo, Jie Zou, Chengfei Yao, Tao Li, Gang Bai
    International Conference on Pattern Recognition 2018
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            init.kaiming_uniform_(m.weight)
            init.zeros_(m.bias)
# ------------------------------------------------------------------------------------------------------------------

    def __init__(self,
                 input_channels,
                 n_classes,
                 patch_size=3,
                 n_planes=90):
        super(Luo3DCNN_KAN_Net, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.n_planes = n_planes
        self.conv1 = nn.Conv3d(1, 90, (24, 3, 3), padding=0, stride=(9, 1, 1))
        self.conv2 = nn.Sequential(
                            FastKANConvLayer(in_channels=1,
                                             out_channels=64,
                                             kernel_size=(3, 3),
                                             stride=(1, 1),
                                             kan_type="RBF"  # "Poly", "Chebyshev", "Fourier", "BSpline"
                                            ),
                            nn.BatchNorm2d(64),
                            nn.ReLU(),
                    )

        self.features_size = self._get_final_flattened_size()
        self.fc_kan = KAN([self.features_size, 512, 512, n_classes],
                          base_activation=torch.nn.ReLU,
                          )

        self.apply(self.weight_init)
# ------------------------------------------------------------------------------------------------------------------

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(
                (1, 1, self.input_channels, self.patch_size, self.patch_size)
            )
            x = self.conv1(x)
            b = x.size(0)
            x = x.view(b, 1, -1, self.n_planes)
            x = self.conv2(x)
            _, c, w, h = x.size()
        return c * w * h
# ------------------------------------------------------------------------------------------------------------------

    def forward(self, x):
        x = F.relu(self.conv1(x))
        b = x.size(0)
        x = x.view(b, 1, -1, self.n_planes)
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.features_size)
        x = self.fc_kan(x)
        return x


class Luo3DCNN_KAN(Model):
    def __init__(self,
                 n_classes,
                 device,
                 n_bands,
                 path_to_weights=None
                 ):
        super(Luo3DCNN_KAN, self).__init__()
        self.hyperparams: dict[str: Any] = dict()
        self.hyperparams['patch_size'] = 3
        self.hyperparams['n_classes'] = n_classes
        self.hyperparams['ignored_labels'] = [0]
        self.hyperparams['device'] = device
        self.hyperparams['n_bands'] = n_bands
        self.hyperparams['center_pixel'] = True
        self.hyperparams['net_name'] = 'hsicnn'
        weights = torch.ones(n_classes)
        weights[torch.LongTensor(self.hyperparams["ignored_labels"])] = 0.0
        weights = weights.to(device)
        self.hyperparams["weights"] = weights

        self.model = Luo3DCNN_KAN_Net(n_bands, n_classes, patch_size=self.hyperparams["patch_size"])

        if path_to_weights:
            self.model.load_state_dict(torch.load(path_to_weights))

        self.hyperparams.setdefault("supervision", "full")
        self.hyperparams.setdefault("flip_augmentation", False)
        self.hyperparams.setdefault("radiation_augmentation", False)
        self.hyperparams.setdefault("mixture_augmentation", False)
        self.hyperparams["center_pixel"] = True
# ------------------------------------------------------------------------------------------------------------------

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            fit_params: Dict):

        fit_params.setdefault('epochs', 10)
        fit_params.setdefault('train_sample_percentage', 0.5)
        fit_params.setdefault('dataloader_mode', 'random')
        fit_params.setdefault('loss', nn.CrossEntropyLoss(weight=self.hyperparams["weights"]))
        fit_params.setdefault('batch_size', 100)
        fit_params.setdefault('optimizer_params', {'learning_rate': 0.01, 'weight_decay': 0.09})
        fit_params.setdefault('optimizer',
                              optim.SGD(self.model.parameters(),
                                        lr=fit_params['optimizer_params']["learning_rate"],
                                        weight_decay=fit_params['optimizer_params']['weight_decay']))
        fit_params.setdefault('scheduler_type', None)
        fit_params.setdefault('scheduler_params', None)

        self.model, history, self.train_mask = super().fit_nn(X=X,
                                                              y=y,
                                                              hyperparams=self.hyperparams,
                                                              model=self.model,
                                                              fit_params=fit_params)
        self.train_loss = history["train_loss"]
        self.val_loss = history["val_loss"]
        self.train_accs = history["train_accuracy"]
        self.val_accs = history["val_accuracy"]
    # ------------------------------------------------------------------------------------------------------------------

    def predict(self,
                X: np.ndarray,
                y: Optional[np.ndarray] = None,
                batch_size=100) -> np.ndarray:

        self.hyperparams.setdefault('batch_size', batch_size)
        prediction = super().predict_nn(X=X,
                                        y=y,
                                        model=self.model,
                                        hyperparams=self.hyperparams)
        return prediction
