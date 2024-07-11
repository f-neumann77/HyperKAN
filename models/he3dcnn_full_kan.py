import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from typing import Any, Dict, Optional

from models.model import Model
from models.kan_layers import KAN, FastKANConv3DLayer


class He3DCNN_KAN_Net(nn.Module):
    """
    MULTI-SCALE 3D DEEP CONVOLUTIONAL NEURAL NETWORK FOR HYPERSPECTRAL
    IMAGE CLASSIFICATION
    Mingyi He, Bo Li, Huahui Chen
    IEEE International Conference on Image Processing (ICIP) 2017
    https://ieeexplore.ieee.org/document/8297014/
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            nn.init.kaiming_uniform(m.weight)
            nn.init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=7):
        super(He3DCNN_KAN_Net, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size

        self.conv1 = FastKANConv3DLayer(1, 16, (11, 3, 3), stride=(3, 1, 1), base_activation=torch.nn.PReLU, grid_size=2)

        self.conv2_1 = FastKANConv3DLayer(16, 16, (1, 1, 1), padding=(0, 0, 0), base_activation=torch.nn.PReLU, grid_size=2)
        self.conv2_2 = FastKANConv3DLayer(16, 16, (3, 1, 1), padding=(1, 0, 0), base_activation=torch.nn.PReLU, grid_size=2)
        self.conv2_3 = FastKANConv3DLayer(16, 16, (5, 1, 1), padding=(2, 0, 0), base_activation=torch.nn.PReLU, grid_size=2)
        self.conv2_4 = FastKANConv3DLayer(16, 16, (11, 1, 1), padding=(5, 0, 0), base_activation=torch.nn.PReLU, grid_size=2)

        self.conv3_1 = FastKANConv3DLayer(16, 16, (1, 1, 1), padding=(0, 0, 0), base_activation=torch.nn.PReLU, grid_size=2)
        self.conv3_2 = FastKANConv3DLayer(16, 16, (3, 1, 1), padding=(1, 0, 0), base_activation=torch.nn.PReLU, grid_size=2)
        self.conv3_3 = FastKANConv3DLayer(16, 16, (5, 1, 1), padding=(2, 0, 0), base_activation=torch.nn.PReLU, grid_size=2)
        self.conv3_4 = FastKANConv3DLayer(16, 16, (11, 1, 1), padding=(5, 0, 0), base_activation=torch.nn.PReLU, grid_size=2)

        self.conv4 = FastKANConv3DLayer(16, 16, (3, 2, 2))

        self.pooling = nn.MaxPool2d((3, 2, 2), stride=(3, 2, 2))
        # the ratio of dropout is 0.6 in our experiments
        self.dropout = nn.Dropout(p=0.6)

        self.features_size = self._get_final_flattened_size()

        self.kan_fc = KAN([self.features_size, 128, 128, n_classes],
                          base_activation=torch.nn.ReLU)

        #self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(
                (1, 1, self.input_channels, self.patch_size, self.patch_size)
            )
            x = self.conv1(x)
            x2_1 = self.conv2_1(x)
            x2_2 = self.conv2_2(x)
            x2_3 = self.conv2_3(x)
            x2_4 = self.conv2_4(x)
            x = x2_1 + x2_2 + x2_3 + x2_4
            x3_1 = self.conv3_1(x)
            x3_2 = self.conv3_2(x)
            x3_3 = self.conv3_3(x)
            x3_4 = self.conv3_4(x)
            x = x3_1 + x3_2 + x3_3 + x3_4
            x = self.conv4(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x2_1 = self.conv2_1(x)
        x2_2 = self.conv2_2(x)
        x2_3 = self.conv2_3(x)
        x2_4 = self.conv2_4(x)
        x = x2_1 + x2_2 + x2_3 + x2_4
        x = nn.functional.relu(x)
        x3_1 = self.conv3_1(x)
        x3_2 = self.conv3_2(x)
        x3_3 = self.conv3_3(x)
        x3_4 = self.conv3_4(x)
        x = x3_1 + x3_2 + x3_3 + x3_4
        x = nn.functional.relu(x)
        x = nn.functional.relu(self.conv4(x))
        x = x.view(-1, self.features_size)
        x = self.dropout(x)
        x = self.kan_fc(x)
        return x


class He3DCNN_KAN(Model):
    def __init__(self,
                 n_classes,
                 device,
                 n_bands,
                 path_to_weights=None
                 ):
        super(He3DCNN_KAN, self).__init__()
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

        self.model = He3DCNN_KAN_Net(n_bands, n_classes, patch_size=self.hyperparams["patch_size"])
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
