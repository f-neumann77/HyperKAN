"""
Based on
https://github.com/IvanDrokin/torch-conv-kan

"""

import math

import torch.nn.functional as F

from typing import Tuple, Union

import torch
import torch.nn as nn
from torch.nn.functional import conv3d, conv2d, conv1d


class WaveletConvND(nn.Module):
    def __init__(self, conv_class, input_dim, output_dim, kernel_size,
                 padding=0, stride=1, dilation=1,
                 ndim: int = 2, wavelet_type='mexican_hat'):
        super(WaveletConvND, self).__init__()

        _shapes = (1, output_dim, input_dim) + tuple(1 for _ in range(ndim))

        self.scale = nn.Parameter(torch.ones(*_shapes))
        self.translation = nn.Parameter(torch.zeros(*_shapes))

        self.ndim = ndim
        self.wavelet_type = wavelet_type

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.wavelet_weights = nn.ModuleList([conv_class(input_dim,
                                                         1,
                                                         kernel_size,
                                                         stride,
                                                         padding,
                                                         dilation,
                                                         groups=1,
                                                         bias=False) for _ in range(output_dim)])

        self.wavelet_out = conv_class(output_dim, output_dim, 1, 1, 0, dilation, groups=1, bias=False)

        for conv_layer in self.wavelet_weights:
            nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='linear')
        nn.init.kaiming_uniform_(self.wavelet_out.weight, nonlinearity='linear')

    @staticmethod
    def _forward_mexican_hat(x):
        term1 = ((x ** 2) - 1)
        term2 = torch.exp(-0.5 * x ** 2)
        wavelet = (2 / (math.sqrt(3) * math.pi ** 0.25)) * term1 * term2
        return wavelet

    @staticmethod
    def _forward_morlet(x):
        omega0 = 5.0  # Central frequency
        real = torch.cos(omega0 * x)
        envelope = torch.exp(-0.5 * x ** 2)
        wavelet = envelope * real
        return wavelet

    @staticmethod
    def _forward_dog(x):
        return -x * torch.exp(-0.5 * x ** 2)

    @staticmethod
    def _forward_meyer(x):
        v = torch.abs(x)
        pi = math.pi

        def meyer_aux(v):
            return torch.where(v <= 1 / 2, torch.ones_like(v),
                               torch.where(v >= 1, torch.zeros_like(v), torch.cos(pi / 2 * nu(2 * v - 1))))

        def nu(t):
            return t ** 4 * (35 - 84 * t + 70 * t ** 2 - 20 * t ** 3)

        # Meyer wavelet calculation using the auxiliary function
        wavelet = torch.sin(pi * v) * meyer_aux(v)
        return wavelet

    def _forward_shannon(self, x):
        pi = math.pi
        sinc = torch.sinc(x / pi)  # sinc(x) = sin(pi*x) / (pi*x)

        _shape = (1, 1, x.size(2)) + tuple(1 for _ in range(self.ndim))
        # Applying a Hamming window to limit the infinite support of the sinc function
        window = torch.hamming_window(x.size(2), periodic=False, dtype=x.dtype,
                                      device=x.device).view(*_shape)
        # Shannon wavelet is the product of the sinc function and the window
        wavelet = sinc * window
        return wavelet

    def forward(self, x):
        x_expanded = x.unsqueeze(1)

        x_scaled = (x_expanded - self.translation) / self.scale

        if self.wavelet_type == 'mexican_hat':
            wavelet = self._forward_mexican_hat(x_scaled)
        elif self.wavelet_type == 'morlet':
            wavelet = self._forward_morlet(x_scaled)
        elif self.wavelet_type == 'dog':
            wavelet = self._forward_dog(x_scaled)
        elif self.wavelet_type == 'meyer':
            wavelet = self._forward_meyer(x_scaled)
        elif self.wavelet_type == 'shannon':
            wavelet = self._forward_shannon(x_scaled)
        else:
            raise ValueError("Unsupported wavelet type")

        wavelet_x = torch.split(wavelet, 1, dim=1)
        output = []
        for group_ind, _x in enumerate(wavelet_x):
            y = self.wavelet_weights[group_ind](_x.squeeze(1))
            # output.append(y.clone())
            output.append(y)
        y = torch.cat(output, dim=1)
        y = self.wavelet_out(y)
        return y


class WaveletConvNDFastPlusOne(WaveletConvND):
    def __init__(self, conv_class, conv_class_d_plus_one, input_dim, output_dim, kernel_size,
                 padding=0, stride=1, dilation=1,
                 ndim: int = 2, wavelet_type='mexican_hat'):
        super(WaveletConvND, self).__init__()

        assert ndim < 3, "fast_plus_one version suppoerts only 1D and 2D convs"

        _shapes = (1, output_dim, input_dim) + tuple(1 for _ in range(ndim))

        self.scale = nn.Parameter(torch.ones(*_shapes))
        self.translation = nn.Parameter(torch.zeros(*_shapes))

        self.ndim = ndim
        self.wavelet_type = wavelet_type

        self.input_dim = input_dim
        self.output_dim = output_dim

        kernel_size_plus = (input_dim,) + kernel_size if isinstance(kernel_size, tuple) else (input_dim,) + (
        kernel_size,) * ndim
        stride_plus = (1,) + stride if isinstance(stride, tuple) else (1,) + (stride,) * ndim
        padding_plus = (0,) + padding if isinstance(padding, tuple) else (0,) + (padding,) * ndim
        dilation_plus = (1,) + dilation if isinstance(dilation, tuple) else (1,) + (dilation,) * ndim

        self.wavelet_weights = conv_class_d_plus_one(output_dim,
                                                     output_dim,
                                                     kernel_size_plus,
                                                     stride_plus,
                                                     padding_plus,
                                                     dilation_plus,
                                                     groups=output_dim,
                                                     bias=False)

        self.wavelet_out = conv_class(output_dim, output_dim, 1, 1, 0, dilation, groups=1, bias=False)

        nn.init.kaiming_uniform_(self.wavelet_weights.weight, nonlinearity='linear')
        nn.init.kaiming_uniform_(self.wavelet_out.weight, nonlinearity='linear')

    def forward(self, x):
        x_expanded = x.unsqueeze(1)

        x_scaled = (x_expanded - self.translation) / self.scale

        if self.wavelet_type == 'mexican_hat':
            wavelet = self._forward_mexican_hat(x_scaled)
        elif self.wavelet_type == 'morlet':
            wavelet = self._forward_morlet(x_scaled)
        elif self.wavelet_type == 'dog':
            wavelet = self._forward_dog(x_scaled)
        elif self.wavelet_type == 'meyer':
            wavelet = self._forward_meyer(x_scaled)
        elif self.wavelet_type == 'shannon':
            wavelet = self._forward_shannon(x_scaled)
        else:
            raise ValueError("Unsupported wavelet type")
        # wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
        # wavelet_output = wavelet_weighted.sum(dim=2)

        y = self.wavelet_weights(wavelet).squeeze(2)
        y = self.wavelet_out(y)
        return y


class WaveletConvNDFast(WaveletConvND):
    def __init__(self, conv_class, input_dim, output_dim, kernel_size,
                 padding=0, stride=1, dilation=1,
                 ndim: int = 2, wavelet_type='mexican_hat'):
        super(WaveletConvND, self).__init__()

        _shapes = (1, output_dim, input_dim) + tuple(1 for _ in range(ndim))

        self.scale = nn.Parameter(torch.ones(*_shapes))
        self.translation = nn.Parameter(torch.zeros(*_shapes))

        self.ndim = ndim
        self.wavelet_type = wavelet_type

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.wavelet_weights = conv_class(output_dim * input_dim,
                                          output_dim,
                                          kernel_size,
                                          stride,
                                          padding,
                                          dilation,
                                          groups=output_dim,
                                          bias=False)

        self.wavelet_out = conv_class(output_dim, output_dim, 1, 1, 0, dilation, groups=1, bias=False)

        nn.init.kaiming_uniform_(self.wavelet_weights.weight, nonlinearity='linear')
        nn.init.kaiming_uniform_(self.wavelet_out.weight, nonlinearity='linear')

    def forward(self, x):
        x_expanded = x.unsqueeze(1)

        x_scaled = (x_expanded - self.translation) / self.scale

        if self.wavelet_type == 'mexican_hat':
            wavelet = self._forward_mexican_hat(x_scaled)
        elif self.wavelet_type == 'morlet':
            wavelet = self._forward_morlet(x_scaled)
        elif self.wavelet_type == 'dog':
            wavelet = self._forward_dog(x_scaled)
        elif self.wavelet_type == 'meyer':
            wavelet = self._forward_meyer(x_scaled)
        elif self.wavelet_type == 'shannon':
            wavelet = self._forward_shannon(x_scaled)
        else:
            raise ValueError("Unsupported wavelet type")
        # wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
        # wavelet_output = wavelet_weighted.sum(dim=2)

        y = self.wavelet_weights(wavelet.flatten(1, 2))
        y = self.wavelet_out(y)
        return y


class WavKANConvNDLayer(nn.Module):
    def __init__(self, conv_class, conv_class_plus1, norm_class, input_dim, output_dim, kernel_size,
                 groups=1, padding=0, stride=1, dilation=1, wav_version: str = 'base',
                 ndim: int = 2, dropout=0.0, wavelet_type='mexican_hat', **norm_kwargs):
        super(WavKANConvNDLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.ndim = ndim
        self.norm_kwargs = norm_kwargs
        assert wavelet_type in ['mexican_hat', 'morlet', 'dog', 'meyer', 'shannon'], \
            ValueError(f"Unsupported wavelet type: {wavelet_type}")
        self.wavelet_type = wavelet_type

        self.dropout = None
        if dropout > 0:
            if ndim == 1:
                self.dropout = nn.Dropout1d(p=dropout)
            if ndim == 2:
                self.dropout = nn.Dropout2d(p=dropout)
            if ndim == 3:
                self.dropout = nn.Dropout3d(p=dropout)
        if groups <= 0:
            raise ValueError('groups must be a positive integer')
        if input_dim % groups != 0:
            raise ValueError('input_dim must be divisible by groups')
        if output_dim % groups != 0:
            raise ValueError('output_dim must be divisible by groups')

        self.base_conv = nn.ModuleList([conv_class(input_dim // groups,
                                                   output_dim // groups,
                                                   kernel_size,
                                                   stride,
                                                   padding,
                                                   dilation,
                                                   groups=1,
                                                   bias=False) for _ in range(groups)])
        if wav_version == 'base':
            self.wavelet_conv = nn.ModuleList(
                [
                    WaveletConvND(
                        conv_class,
                        input_dim // groups,
                        output_dim // groups,
                        kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                        ndim=ndim, wavelet_type=wavelet_type
                    ) for _ in range(groups)
                ]
            )
        elif wav_version == 'fast':
            self.wavelet_conv = nn.ModuleList(
                [
                    WaveletConvNDFast(
                        conv_class,
                        input_dim // groups,
                        output_dim // groups,
                        kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                        ndim=ndim, wavelet_type=wavelet_type
                    ) for _ in range(groups)
                ]
            )
        elif wav_version == 'fast_plus_one':

            self.wavelet_conv = nn.ModuleList(
                [
                    WaveletConvNDFastPlusOne(
                        conv_class, conv_class_plus1,
                        input_dim // groups,
                        output_dim // groups,
                        kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                        ndim=ndim, wavelet_type=wavelet_type
                    ) for _ in range(groups)
                ]
            )

        self.layer_norm = nn.ModuleList([norm_class(output_dim // groups, **norm_kwargs) for _ in range(groups)])

        self.base_activation = nn.SiLU()

    def forward_wavkan(self, x, group_ind):
        # You may like test the cases like Spl-KAN
        base_output = self.base_conv[group_ind](self.base_activation(x))

        if self.dropout is not None:
            x = self.dropout(x)

        wavelet_output = self.wavelet_conv[group_ind](x)

        combined_output = wavelet_output + base_output

        # Apply batch normalization
        return self.layer_norm[group_ind](combined_output)

    def forward(self, x):
        split_x = torch.split(x, self.inputdim // self.groups, dim=1)
        output = []
        for group_ind, _x in enumerate(split_x):
            y = self.forward_wavkan(_x, group_ind)
            output.append(y.clone())
        y = torch.cat(output, dim=1)
        return y


class WavKANConv3DLayer(WavKANConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, groups=1, padding=0, stride=1, dilation=1,
                 dropout=0.0, wavelet_type='mexican_hat', norm_layer=nn.BatchNorm3d,
                 wav_version: str = 'fast', **norm_kwargs):
        super(WavKANConv3DLayer, self).__init__(nn.Conv3d, None, norm_layer, input_dim, output_dim, kernel_size,
                                                groups=groups, padding=padding, stride=stride, dilation=dilation,
                                                ndim=3, dropout=dropout, wavelet_type=wavelet_type,
                                                wav_version=wav_version, **norm_kwargs)


class WavKANConv2DLayer(WavKANConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, groups=1, padding=0, stride=1, dilation=1,
                 dropout=0.0, wavelet_type='mexican_hat', norm_layer=nn.BatchNorm2d,
                 wav_version: str = 'fast', **norm_kwargs):
        super(WavKANConv2DLayer, self).__init__(nn.Conv2d, nn.Conv3d, norm_layer, input_dim, output_dim, kernel_size,
                                                groups=groups, padding=padding, stride=stride, dilation=dilation,
                                                ndim=2, dropout=dropout, wavelet_type=wavelet_type,
                                                wav_version=wav_version, **norm_kwargs)


class WavKANConv1DLayer(WavKANConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, groups=1, padding=0, stride=1, dilation=1,
                 dropout=0.0, wavelet_type='mexican_hat', norm_layer=nn.BatchNorm1d,
                 wav_version: str = 'fast', **norm_kwargs):
        super(WavKANConv1DLayer, self).__init__(nn.Conv1d, nn.Conv2d, norm_layer, input_dim, output_dim, kernel_size,
                                                groups=groups, padding=padding, stride=stride, dilation=dilation,
                                                ndim=1, dropout=dropout, wavelet_type=wavelet_type,
                                                wav_version=wav_version, **norm_kwargs)


class KABNConvNDLayer(nn.Module):
    def __init__(self, conv_class, norm_class, conv_w_fun, input_dim, output_dim, degree, kernel_size,
                 groups=1, padding=0, stride=1, dilation=1, dropout: float = 0.0,
                 ndim: int = 2, **norm_kwargs):
        super(KABNConvNDLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.base_activation = nn.SiLU()
        self.conv_w_fun = conv_w_fun
        self.ndim = ndim
        self.dropout = None
        self.norm_kwargs = norm_kwargs
        if dropout > 0:
            if ndim == 1:
                self.dropout = nn.Dropout1d(p=dropout)
            if ndim == 2:
                self.dropout = nn.Dropout2d(p=dropout)
            if ndim == 3:
                self.dropout = nn.Dropout3d(p=dropout)

        if groups <= 0:
            raise ValueError('groups must be a positive integer')
        if input_dim % groups != 0:
            raise ValueError('input_dim must be divisible by groups')
        if output_dim % groups != 0:
            raise ValueError('output_dim must be divisible by groups')

        self.base_conv = nn.ModuleList([conv_class(input_dim // groups,
                                                   output_dim // groups,
                                                   kernel_size,
                                                   stride,
                                                   padding,
                                                   dilation,
                                                   groups=1,
                                                   bias=False) for _ in range(groups)])

        self.layer_norm = nn.ModuleList([norm_class(output_dim // groups, **norm_kwargs) for _ in range(groups)])

        poly_shape = (groups, output_dim // groups, (input_dim // groups) * (degree + 1)) + tuple(
            kernel_size for _ in range(ndim))

        self.poly_weights = nn.Parameter(torch.randn(*poly_shape))

        # Initialize weights using Kaiming uniform distribution for better training start
        for conv_layer in self.base_conv:
            nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='linear')

        nn.init.kaiming_uniform_(self.poly_weights, nonlinearity='linear')


    def bernstein_poly(self, x, degree):

        bernsteins = torch.ones(x.shape + (self.degree + 1, ), dtype=x.dtype, device=x.device)
        for j in range(1, degree + 1):
            for k in range(degree + 1 - j):
                bernsteins[..., k] = bernsteins[..., k] * (1 - x) + bernsteins[..., k + 1] * x

        bernsteins = bernsteins.moveaxis(-1, 2)
        bernsteins = bernsteins.flatten(1, 2)

        return bernsteins

    def forward_kab(self, x, group_index):
        # Apply base activation to input and then linear transform with base weights
        base_output = self.base_conv[group_index](x)

        # Normalize x to the range [-1, 1] for stable Legendre polynomial computation
        x_normalized = torch.sigmoid(x)

        if self.dropout is not None:
            x_normalized = self.dropout(x_normalized)

        # Compute Legendre polynomials for the normalized x
        bernstein_basis = self.bernstein_poly(x_normalized, self.degree)
        # Reshape legendre_basis to match the expected input dimensions for linear transformation
        # Compute polynomial output using polynomial weights
        poly_output = self.conv_w_fun(bernstein_basis, self.poly_weights[group_index],
                                      stride=self.stride, dilation=self.dilation,
                                      padding=self.padding, groups=1)

        # poly_output = poly_output.view(orig_shape[0], orig_shape[1], orig_shape[2], orig_shape[3], self.outdim // self.groups)
        # Combine base and polynomial outputs, normalize, and activate
        x = base_output + poly_output
        if isinstance(self.layer_norm[group_index], nn.LayerNorm):
            orig_shape = x.shape
            x = self.layer_norm[group_index](x.view(orig_shape[0], -1)).view(orig_shape)
        else:
            x = self.layer_norm[group_index](x)
        x = self.base_activation(x)

        return x

    def forward(self, x):

        split_x = torch.split(x, self.inputdim // self.groups, dim=1)
        output = []
        for group_ind, _x in enumerate(split_x):
            y = self.forward_kab(_x, group_ind)
            output.append(y.clone())
        y = torch.cat(output, dim=1)
        return y


class KABNConv3DLayer(KABNConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, degree=3, groups=1, padding=0, stride=1, dilation=1,
                 dropout: float = 0.0, norm_layer=nn.InstanceNorm3d, **norm_kwargs):
        super(KABNConv3DLayer, self).__init__(nn.Conv3d, norm_layer, conv3d,
                                              input_dim, output_dim,
                                              degree, kernel_size,
                                              groups=groups, padding=padding, stride=stride, dilation=dilation,
                                              ndim=3, dropout=dropout, **norm_kwargs)


class KABNConv2DLayer(KABNConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, degree=3, groups=1, padding=0, stride=1, dilation=1,
                 dropout: float = 0.0, norm_layer=nn.InstanceNorm2d, **norm_kwargs):
        super(KABNConv2DLayer, self).__init__(nn.Conv2d, norm_layer, conv2d,
                                              input_dim, output_dim,
                                              degree, kernel_size,
                                              groups=groups, padding=padding, stride=stride, dilation=dilation,
                                              ndim=2, dropout=dropout, **norm_kwargs)


class KABNConv1DLayer(KABNConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, degree=3, groups=1, padding=0, stride=1, dilation=1,
                 dropout: float = 0.0, norm_layer=nn.InstanceNorm1d, **norm_kwargs):
        super(KABNConv1DLayer, self).__init__(nn.Conv1d, norm_layer, conv1d,
                                              input_dim, output_dim,
                                              degree, kernel_size,
                                              groups=groups, padding=padding, stride=stride, dilation=dilation,
                                              ndim=1, dropout=dropout, **norm_kwargs)


class FastKANConvNDLayer(nn.Module):
    def __init__(self, conv_class, norm_class, input_dim, output_dim, kernel_size,
                 groups=1, padding=0, stride=1, dilation=1,
                 ndim: int = 2, grid_size=8, base_activation=nn.SiLU, grid_range=[-2, 2], dropout=0.0, **norm_kwargs):
        super(FastKANConvNDLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.ndim = ndim
        self.grid_size = grid_size
        self.base_activation = base_activation()
        self.grid_range = grid_range
        self.norm_kwargs = norm_kwargs

        if groups <= 0:
            raise ValueError('groups must be a positive integer')
        if input_dim % groups != 0:
            raise ValueError('input_dim must be divisible by groups')
        if output_dim % groups != 0:
            raise ValueError('output_dim must be divisible by groups')

        self.base_conv = nn.ModuleList([conv_class(input_dim // groups,
                                                   output_dim // groups,
                                                   kernel_size,
                                                   stride,
                                                   padding,
                                                   dilation,
                                                   groups=1,
                                                   bias=False) for _ in range(groups)])

        self.spline_conv = nn.ModuleList([conv_class(grid_size * input_dim // groups,
                                                     output_dim // groups,
                                                     kernel_size,
                                                     stride,
                                                     padding,
                                                     dilation,
                                                     groups=1,
                                                     bias=False) for _ in range(groups)])

        self.layer_norm = nn.ModuleList([norm_class(input_dim // groups, **norm_kwargs) for _ in range(groups)])

        self.rbf = RadialBasisFunction(grid_range[0], grid_range[1], grid_size)

        self.dropout = None
        if dropout > 0:
            if ndim == 1:
                self.dropout = nn.Dropout1d(p=dropout)
            if ndim == 2:
                self.dropout = nn.Dropout2d(p=dropout)
            if ndim == 3:
                self.dropout = nn.Dropout3d(p=dropout)

        # Initialize weights using Kaiming uniform distribution for better training start
        for conv_layer in self.base_conv:
            nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='linear')

        for conv_layer in self.spline_conv:
            nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='linear')

    def forward_fast_kan(self, x, group_index):

        # Apply base activation to input and then linear transform with base weights
        base_output = self.base_conv[group_index](self.base_activation(x))
        if self.dropout is not None:
            x = self.dropout(x)
        spline_basis = self.rbf(self.layer_norm[group_index](x))
        spline_basis = spline_basis.moveaxis(-1, 2).flatten(1, 2)
        spline_output = self.spline_conv[group_index](spline_basis)
        x = base_output + spline_output

        return x

    def forward(self, x):
        split_x = torch.split(x, self.inputdim // self.groups, dim=1)
        output = []
        for group_ind, _x in enumerate(split_x):
            y = self.forward_fast_kan(_x, group_ind)
            output.append(y.clone())
        y = torch.cat(output, dim=1)
        return y


class FastKANConv3DLayer(FastKANConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, groups=1, padding=0, stride=1, dilation=1,
                 grid_size=8, base_activation=nn.SiLU, grid_range=[-2, 2], dropout=0.0,
                 norm_layer=nn.InstanceNorm3d, **norm_kwargs):
        super(FastKANConv3DLayer, self).__init__(nn.Conv3d, norm_layer,
                                                 input_dim, output_dim,
                                                 kernel_size,
                                                 groups=groups, padding=padding, stride=stride, dilation=dilation,
                                                 ndim=3,
                                                 grid_size=grid_size, base_activation=base_activation,
                                                 grid_range=grid_range,
                                                 dropout=dropout, **norm_kwargs)


class FastKANConv2DLayer(FastKANConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, groups=1, padding=0, stride=1, dilation=1,
                 grid_size=8, base_activation=nn.SiLU, grid_range=[-2, 2], dropout=0.0,
                 norm_layer=nn.InstanceNorm2d, **norm_kwargs):
        super(FastKANConv2DLayer, self).__init__(nn.Conv2d, norm_layer,
                                                 input_dim, output_dim,
                                                 kernel_size,
                                                 groups=groups, padding=padding, stride=stride, dilation=dilation,
                                                 ndim=2,
                                                 grid_size=grid_size, base_activation=base_activation,
                                                 grid_range=grid_range,
                                                 dropout=dropout, **norm_kwargs)


class FastKANConv1DLayer(FastKANConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, groups=1, padding=0, stride=1, dilation=1,
                 grid_size=8, base_activation=nn.SiLU, grid_range=[-2, 2], dropout=0.0,
                 norm_layer=nn.InstanceNorm1d, **norm_kwargs):
        super(FastKANConv1DLayer, self).__init__(nn.Conv1d, norm_layer,
                                                 input_dim, output_dim,
                                                 kernel_size,
                                                 groups=groups, padding=padding, stride=stride, dilation=dilation,
                                                 ndim=1,
                                                 grid_size=grid_size, base_activation=base_activation,
                                                 grid_range=grid_range,
                                                 dropout=dropout, **norm_kwargs)


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


class KANLinear(torch.nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            enable_standalone_scale_spline=True,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                    torch.arange(-spline_order, grid_size + spline_order + 1) * h
                    + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                    (
                            torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                            - 1 / 2
                    )
                    * self.scale_noise
                    / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order: -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                            (x - grid[:, : -(k + 1)])
                            / (grid[:, k:-1] - grid[:, : -(k + 1)])
                            * bases[:, :, :-1]
                    ) + (
                            (grid[:, k + 1:] - x)
                            / (grid[:, k + 1:] - grid[:, 1:(-k)])
                            * bases[:, :, 1:]
                    )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
                torch.arange(
                    self.grid_size + 1, dtype=torch.float32, device=x.device
                ).unsqueeze(1)
                * uniform_step
                + x_sorted[0]
                - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
                regularize_activation * regularization_loss_activation
                + regularize_entropy * regularization_loss_entropy
        )


class KAN(torch.nn.Module):
    def __init__(
            self,
            layers_hidden,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                ))

            self.layers.append(torch.nn.BatchNorm1d(out_features))

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )
