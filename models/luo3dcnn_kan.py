from typing import Any, Optional, Dict, Union, Tuple
import numpy as np

from models.kan_linear import KAN
from models.model import Model

import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


class PolynomialFunction(nn.Module):
    def __init__(self,
                 degree: int = 3):
        super().__init__()
        self.degree = degree

    def forward(self, x):
        return torch.stack([x ** i for i in range(self.degree)], dim=-1)


class BSplineFunction(nn.Module):
    def __init__(self, grid_min: float = -2.,
                 grid_max: float = 2., degree: int = 3, num_basis: int = 8):
        super(BSplineFunction, self).__init__()
        self.degree = degree
        self.num_basis = num_basis
        self.knots = torch.linspace(grid_min, grid_max, num_basis + degree + 1)  # Uniform knots

    def basis_function(self, i, k, t):
        if k == 0:
            return ((self.knots[i] <= t) & (t < self.knots[i + 1])).float()
        else:
            left_num = (t - self.knots[i]) * self.basis_function(i, k - 1, t)
            left_den = self.knots[i + k] - self.knots[i]
            left = left_num / left_den if left_den != 0 else 0

            right_num = (self.knots[i + k + 1] - t) * self.basis_function(i + 1, k - 1, t)
            right_den = self.knots[i + k + 1] - self.knots[i + 1]
            right = right_num / right_den if right_den != 0 else 0

            return left + right

    def forward(self, x):
        x = x.squeeze()  # Assuming x is of shape (B, 1)
        basis_functions = torch.stack([self.basis_function(i, self.degree, x) for i in range(self.num_basis)], dim=-1)
        return basis_functions


class ChebyshevFunction(nn.Module):
    def __init__(self, degree: int = 4):
        super(ChebyshevFunction, self).__init__()
        self.degree = degree

    def forward(self, x):
        chebyshev_polynomials = [torch.ones_like(x), x]
        for n in range(2, self.degree):
            chebyshev_polynomials.append(2 * x * chebyshev_polynomials[-1] - chebyshev_polynomials[-2])
        return torch.stack(chebyshev_polynomials, dim=-1)


class FourierBasisFunction(nn.Module):
    def __init__(self,
                 num_frequencies: int = 4,
                 period: float = 1.0):
        super(FourierBasisFunction, self).__init__()
        assert num_frequencies % 2 == 0, "num_frequencies must be even"
        self.num_frequencies = num_frequencies
        self.period = nn.Parameter(torch.Tensor([period]), requires_grad=False)

    def forward(self, x):
        frequencies = torch.arange(1, self.num_frequencies // 2 + 1, device=x.device)
        sin_components = torch.sin(2 * torch.pi * frequencies * x[..., None] / self.period)
        cos_components = torch.cos(2 * torch.pi * frequencies * x[..., None] / self.period)
        basis_functions = torch.cat([sin_components, cos_components], dim=-1)
        return basis_functions


class RadialBasisFunction(nn.Module):
    def __init__(
            self,
            grid_min: float = -2.,
            grid_max: float = 2.,
            num_grids: int = 4,
            denominator: float = None,
    ):
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = torch.nn.Parameter(grid, requires_grad=False)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

    def forward(self, x):
        return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)


class SplineConv2D(nn.Conv2d):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]] = 3,
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 init_scale: float = 0.1,
                 padding_mode: str = "zeros",
                 **kw
                 ) -> None:
        self.init_scale = init_scale
        super().__init__(in_channels,
                         out_channels,
                         kernel_size,
                         stride,
                         padding,
                         dilation,
                         groups,
                         bias,
                         padding_mode,
                         **kw
                         )

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class FastKANConvLayer(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]] = 3,
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 grid_min: float = -2.,
                 grid_max: float = 2.,
                 num_grids: int = 4,
                 use_base_update: bool = True,
                 base_activation=F.relu,  # silu
                 spline_weight_init_scale: float = 0.1,
                 padding_mode: str = "zeros",
                 kan_type: str = "RBF",
                 ) -> None:

        super().__init__()
        if kan_type == "RBF":
            self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
        elif kan_type == "Fourier":
            self.rbf = FourierBasisFunction(num_grids)
        elif kan_type == "Poly":
            self.rbf = PolynomialFunction(num_grids)
        elif kan_type == "Chebyshev":
            self.rbf = ChebyshevFunction(num_grids)
        elif kan_type == "BSpline":
            self.rbf = BSplineFunction(grid_min, grid_max, 3, num_grids)

        self.spline_conv = SplineConv2D(in_channels * num_grids,
                                        out_channels,
                                        kernel_size,
                                        stride,
                                        padding,
                                        dilation,
                                        groups,
                                        bias,
                                        spline_weight_init_scale,
                                        padding_mode)

        self.use_base_update = use_base_update
        if use_base_update:
            self.base_activation = base_activation
            self.base_conv = nn.Conv2d(in_channels,
                                       out_channels,
                                       kernel_size,
                                       stride,
                                       padding,
                                       dilation,
                                       groups,
                                       bias,
                                       padding_mode)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        x_rbf = self.rbf(x.view(batch_size, channels, -1)).view(batch_size, channels, height, width, -1)
        x_rbf = x_rbf.permute(0, 4, 1, 2, 3).contiguous().view(batch_size, -1, height, width)

        # Apply spline convolution
        ret = self.spline_conv(x_rbf)

        if self.use_base_update:
            base = self.base_conv(self.base_activation(x))
            ret = ret + base

        return ret


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
    # ------------------------------------------------------------------------------------------------------------------
