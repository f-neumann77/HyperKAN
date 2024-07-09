import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.nn import init
from typing import Any, Optional, Dict

from models.model import Model


class MLP_Net(nn.Module):
    """
    Baseline network
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, dropout=False):
        super(MLP_Net, self).__init__()
        self.use_dropout = dropout
        if dropout:
            self.dropout = nn.Dropout(p=0.15)

        self.fc1 = nn.Linear(input_channels, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, n_classes)

        self.apply(self.weight_init)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = F.relu(self.fc2(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = F.relu(self.fc3(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = self.fc4(x)
        return x


class MLP(Model):
    def __init__(self,
                 n_classes,
                 device,
                 n_bands,
                 path_to_weights=None
                 ):
        super(MLP, self).__init__()
        self.hyperparams: dict[str: Any] = dict()
        self.hyperparams['patch_size'] = 1
        self.hyperparams['n_classes'] = n_classes
        self.hyperparams['ignored_labels'] = [0]
        self.hyperparams['device'] = device
        self.hyperparams['n_bands'] = n_bands
        self.hyperparams['net_name'] = 'nn'

        weights = torch.ones(n_classes)
        weights[torch.LongTensor(self.hyperparams["ignored_labels"])] = 0.0
        weights = weights.to(device)
        self.hyperparams["weights"] = weights

        self.model = MLP_Net(n_bands, n_classes, dropout=True)

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
        fit_params.setdefault('loss', nn.CrossEntropyLoss(weight=self.hyperparams["weights"]))  # TODO custom loss?
        fit_params.setdefault('batch_size', 100)
        fit_params.setdefault('optimizer_params', {'learning_rate': 0.0001, 'weight_decay': 0.0005})
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
