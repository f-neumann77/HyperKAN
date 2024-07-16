import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from typing import Any, Dict, Optional

from models.model import Model


class M2DCNN_Net(nn.Module):
    @staticmethod
    def weight_init(m):
        # [All the trainable parameters in our CNN should be initialized to
        # be a random value between −0.05 and 0.05.]
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.uniform_(m.weight, -0.05, 0.05)
            nn.init.zeros_(m.bias)
    # ------------------------------------------------------------------------------------------------------------------

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(1, self.input_channels, self.patch_size, self.patch_size)
            x = self.conv_1(x)
            x = self.conv_2(x)
            _, c, w, h = x.size()
        return c * w * h
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self,
                 input_channels,
                 n_classes,
                 patch_size=5):
        super(M2DCNN_Net, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        C1 = 3 * self.input_channels
        self.conv_1 = nn.Conv2d(input_channels, C1, (3, 3))
        self.conv_2 = nn.Conv2d(C1, 3 * C1, (3, 3))
        self.do_1 = nn.Dropout(0.25)
        self.features_size = self._get_final_flattened_size()
        self.fc_1 = nn.Linear(self.features_size, 6 * self.input_channels)
        self.do_2 = nn.Dropout(0.5)
        self.fc_2 = nn.Linear(6 * self.input_channels, n_classes)
    # ------------------------------------------------------------------------------------------------------------------

    def forward(self, x):
        x = self.conv_1(x)
        x = nn.functional.relu(x)
        x = self.conv_2(x)
        x = self.do_1(x)
        x = nn.functional.relu(x)
        x = x.view(-1, self.features_size)
        x = self.fc_1(x)
        x = nn.functional.relu(x)
        x = self.do_2(x)
        x = self.fc_2(x)
        return x
# ----------------------------------------------------------------------------------------------------------------------


class M2DCNN(Model):
    def __init__(self,
                 n_classes,
                 device,
                 n_bands,
                 path_to_weights=None
                 ):
        super(M2DCNN, self).__init__()
        self.hyperparams: dict[str: Any] = dict()
        self.hyperparams['patch_size'] = 5
        self.hyperparams['n_classes'] = n_classes
        self.hyperparams['ignored_labels'] = [0]
        self.hyperparams['device'] = device
        self.hyperparams['n_bands'] = n_bands
        self.hyperparams['center_pixel'] = True
        self.hyperparams['net_name'] = 'm2dcnn'
        self.hyperparams['is_3d'] = False
        weights = torch.ones(n_classes)
        weights[torch.LongTensor(self.hyperparams["ignored_labels"])] = 0.0
        weights = weights.to(device)
        self.hyperparams["weights"] = weights

        self.model = M2DCNN_Net(n_bands,
                                n_classes,
                                patch_size=self.hyperparams['patch_size'])

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