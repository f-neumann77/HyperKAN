import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from typing import *

from models.model import Model
from models.kan_layers import KAN


class M1DCNN_KAN_Net(nn.Module):
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            nn.init.uniform_(m.weight, -0.05, 0.05)
            nn.init.zeros_(m.bias)
    # ------------------------------------------------------------------------------------------------------------------

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, self.input_channels)
            x = self.conv_1(x)
            x = self.conv_2(x)
            x = self.pool(x)
        return x.numel()
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self,
                 input_channels,
                 n_classes,
                 kernel_size=None,
                 pool_size=None):
        super(M1DCNN_KAN_Net, self).__init__()
        if kernel_size is None:
            kernel_size = math.ceil(input_channels / 9)
        if pool_size is None:
            pool_size = math.ceil(kernel_size / 5)
        self.input_channels = input_channels

        self.conv_1 = nn.Conv1d(1, 20, kernel_size)
        self.conv_2 = nn.Conv1d(20, 20, kernel_size)
        self.bn_conv = nn.BatchNorm1d(20)
        self.pool = nn.MaxPool1d(pool_size)
        self.features_size = self._get_final_flattened_size()
        self.kan_fc = KAN([self.features_size, 512, 512, n_classes],
                           base_activation=torch.nn.ReLU,
                              )
    # ------------------------------------------------------------------------------------------------------------------

    def forward(self, x):
        # [In our design architecture, we choose the hyperbolic tangent function tanh(u)]
        x = x.squeeze(dim=-1).squeeze(dim=-1)
        x = x.unsqueeze(1)
        x = self.conv_1(x)
        x = self.bn_conv(x)
        x = self.conv_2(x)
        x = torch.tanh(self.pool(x))
        x = x.view(-1, self.features_size)
        x = self.kan_fc(x)
        return x
# ----------------------------------------------------------------------------------------------------------------------


class M1DCNN_KAN(Model):
    def __init__(self,
                 n_classes,
                 device,
                 n_bands,
                 path_to_weights=None
                 ):
        super(M1DCNN_KAN, self).__init__()
        self.hyperparams: dict[str: Any] = dict()
        self.hyperparams['patch_size'] = 1
        self.hyperparams['n_classes'] = n_classes
        self.hyperparams['ignored_labels'] = [0]
        self.hyperparams['device'] = device
        self.hyperparams['n_bands'] = n_bands
        self.hyperparams['center_pixel'] = True
        self.hyperparams['net_name'] = 'm1dcnn'
        weights = torch.ones(n_classes)
        weights[torch.LongTensor(self.hyperparams["ignored_labels"])] = 0.0
        weights = weights.to(device)
        self.hyperparams["weights"] = weights

        self.model = M1DCNN_KAN_Net(n_bands, n_classes)

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
        fit_params.setdefault('optimizer_params', {'learning_rate': 0.05})
        fit_params.setdefault('optimizer',
                              optim.SGD(self.model.parameters(),
                                        lr=fit_params['optimizer_params']["learning_rate"]))
        fit_params.setdefault('scheduler_type', None)
        fit_params.setdefault('scheduler_params', None)

        fit_params.setdefault('wandb', self.wandb_run)
        fit_params.setdefault('tensorboard', self.writer)

        fit_params.setdefault('wandb_vis', False)
        fit_params.setdefault('tensorboard_viz', False)

        self.model, history, self.train_mask = super().fit_nn(X=X,
                                             y=y,
                                             hyperparams=self.hyperparams,
                                             model=self.model,
                                             fit_params=fit_params)
        self.train_loss = history["train_loss"]
        self.val_loss = history["val_loss"]
        self.train_accs = history["train_accuracy"]
        self.val_accs = history["val_accuracy"]
        self.lrs = history["lr"]
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
# ----------------------------------------------------------------------------------------------------------------------
