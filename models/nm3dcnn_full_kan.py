import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from typing import Any, Dict, Optional

from models.model import Model
from models.kan_layers import KAN, FastKANConv1DLayer, FastKANConv3DLayer


class ParallelConvBlock(nn.Module):
    def __init__(self,
                 inp,
                 out):
        super().__init__()
        self.kaconv_1 = FastKANConv3DLayer(inp, 8, (1, 1, 1), padding=(0, 0, 0), base_activation=torch.nn.PReLU, grid_size=2)
        self.bn_conv1 = nn.BatchNorm3d(8)
        self.kaconv_2 = FastKANConv3DLayer(8, 8, (3, 1, 1), padding=(1, 0, 0), base_activation=torch.nn.PReLU, grid_size=2)
        self.bn_conv2 = nn.BatchNorm3d(8)
        self.kaconv_3 = FastKANConv3DLayer(8, 8, (5, 1, 1), padding=(2, 0, 0), base_activation=torch.nn.PReLU, grid_size=2)
        self.bn_conv3 = nn.BatchNorm3d(8)
        self.kaconv_4 = FastKANConv3DLayer(8, out, (11, 1, 1), padding=(5, 0, 0), base_activation=torch.nn.PReLU, grid_size=2)
        self.bn_conv4 = nn.BatchNorm3d(out)

    def forward(self,
                x,
                **kwargs):

        x1 = self.kaconv_1(x)
        x2 = self.kaconv_2(x)
        x3 = self.kaconv_3(x)
        x4 = self.kaconv_4(x)

        return x1 + x2 + x3 + x4
# ----------------------------------------------------------------------------------------------------------------------


class NM3DCNN_KAN_Net(nn.Module):
    """
    MULTI-SCALE 3D DEEP CONVOLUTIONAL NEURAL NETWORK FOR HYPERSPECTRAL
    IMAGE CLASSIFICATION
    Mingyi He, Bo Li, Huahui Chen
    IEEE International Conference on Image Processing (ICIP) 2017
    https://ieeexplore.ieee.org/document/8297014/

    modified by N.A. Firsov, A.V. Nikonorov
    DOI: 10.18287/2412-6179-CO-1038
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            nn.init.kaiming_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self,
                 input_channels,
                 n_classes,
                 patch_size=7):
        super(NM3DCNN_KAN_Net, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size

        self.kaconv1 = FastKANConv3DLayer(1, 8, (11, 3, 3), stride=(3, 1, 1), base_activation=torch.nn.PReLU, grid_size=2)
        self.bn_conv1 = nn.BatchNorm3d(8)

        self.pcb_1 = ParallelConvBlock(inp=8, out=8)
        self.bn_pcb_1 = nn.BatchNorm3d(8)

        self.pcb_2 = ParallelConvBlock(inp=8, out=8)
        self.bn_pcb_2 = nn.BatchNorm3d(8)

        self.kaconv4 = FastKANConv3DLayer(8, 8, (3, 2, 2), base_activation=torch.nn.PReLU, grid_size=2)
        self.bn_conv4 = nn.BatchNorm3d(8)

        self.features_size = self._get_final_flattened_size()

        self.kan_fc = KAN([self.features_size, 128, 128, n_classes],
                          base_activation=torch.nn.PReLU)

        #self.apply(self.weight_init)
    # ------------------------------------------------------------------------------------------------------------------

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(
                (1, 1, self.input_channels, self.patch_size, self.patch_size)
            )
            x = self.kaconv1(x)
            x = self.bn_conv1(x)

            x = self.pcb_1(x)
            x = self.bn_pcb_1(x)

            x = self.pcb_2(x)
            x = self.bn_pcb_2(x)

            x = self.kaconv4(x)
            x = self.bn_conv4(x)

            _, t, c, w, h = x.size()
        return t * c * w * h
    # ------------------------------------------------------------------------------------------------------------------

    def forward(self, x):

        x = self.kaconv1(x)
        x = self.bn_conv1(x)
        x = nn.functional.relu(x)
        x = self.pcb_1(x)
        x = nn.functional.relu(x)
        x = self.pcb_2(x)
        x = nn.functional.relu(x)

        x = self.kaconv4(x)
        x = self.bn_conv4(x)
        x = nn.functional.relu(x)

        x = x.view(-1, self.features_size)
        x = self.kan_fc(x)
        return x
# ----------------------------------------------------------------------------------------------------------------------


class NM3DCNN_KAN(Model):
    def __init__(self,
                 n_classes,
                 device,
                 n_bands,
                 path_to_weights=None
                 ):
        super(NM3DCNN_KAN, self).__init__()
        self.hyperparams: dict[str: Any] = dict()
        self.hyperparams['patch_size'] = 7
        self.hyperparams['n_bands'] = n_bands
        self.hyperparams['net_name'] = 'nm3dcnn'
        self.hyperparams['n_classes'] = n_classes
        self.hyperparams['ignored_labels'] = [0]
        self.hyperparams['device'] = device

        weights = torch.ones(n_classes)
        weights[torch.LongTensor(self.hyperparams["ignored_labels"])] = 0.0
        weights = weights.to(device)
        self.hyperparams["weights"] = weights

        self.model = NM3DCNN_KAN_Net(n_bands, n_classes, patch_size=self.hyperparams["patch_size"])
        # For Adagrad, we need to load the model on GPU before creating the optimizer
        self.model = self.model.to(device)

        self.hyperparams.setdefault("supervision", "full")
        self.hyperparams.setdefault("flip_augmentation", False)
        self.hyperparams.setdefault("radiation_augmentation", False)
        self.hyperparams.setdefault("mixture_augmentation", False)
        self.hyperparams["center_pixel"] = True

        if path_to_weights:
            self.model.load_state_dict(torch.load(path_to_weights))
    # ------------------------------------------------------------------------------------------------------------------

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            fit_params: Dict):

        fit_params.setdefault('epochs', 10)
        fit_params.setdefault('train_sample_percentage', 0.5)
        fit_params.setdefault('dataloader_mode', 'random')
        fit_params.setdefault('loss', nn.CrossEntropyLoss(weight=self.hyperparams["weights"]))
        fit_params.setdefault('batch_size', 40)
        fit_params.setdefault('optimizer_params', {'learning_rate': 0.01, 'weight_decay': 0.01})
        fit_params.setdefault('optimizer',
                              optim.SGD(self.model.parameters(),
                                        lr=fit_params['optimizer_params']["learning_rate"],
                                        weight_decay=fit_params['optimizer_params']['weight_decay']))
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
                batch_size=40) -> np.ndarray:

        self.hyperparams.setdefault('batch_size', batch_size)
        prediction = super().predict_nn(X=X,
                                        y=y,
                                        model=self.model,
                                        hyperparams=self.hyperparams)

        return prediction
# ----------------------------------------------------------------------------------------------------------------------
