import numpy as np
import torch
import torch.optim as optim

from einops import rearrange
from torch import nn
from typing import Any, Dict, Optional, Union

from dataloaders.torch_dataloader import create_torch_loader
from dataloaders.utils import get_dataset, sample_gt
from models.model import train, save_train_mask, Model
from models.utils import camel_to_snake


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
        nn.init.kaiming_normal_(m.weight)
# ----------------------------------------------------------------------------------------------------------------------


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x
# ----------------------------------------------------------------------------------------------------------------------


# Equivalent to PreNorm
class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
# ----------------------------------------------------------------------------------------------------------------------


# Equivalent to FeedForward
class MLP_Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
# ----------------------------------------------------------------------------------------------------------------------


class Attention(nn.Module):

    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5  # 1/sqrt(dim)

        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)  # Wq,Wk,Wv for each vector, that's why *3

        self.nn1 = nn.Linear(dim, dim)

        self.do1 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        b, n, _ = x.shape
        h = self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)  # split into multi head attentions

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask:
            mask = nn.functional.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)  # follow the softmax,q,d,v equation in the paper

        out = torch.einsum('bhij,bhjd->bhid', attn, v)  # product of v times whatever inside softmax
        out = rearrange(out, 'b h n d -> b n (h d)')  # concat heads into one matrix, ready for next encoder block
        out = self.nn1(out)
        out = self.do1(out)
        return out
# ----------------------------------------------------------------------------------------------------------------------


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(LayerNormalize(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(LayerNormalize(dim, MLP_Block(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attention, mlp in self.layers:
            x = attention(x, mask=mask)  # go to attention
            x = mlp(x)  # go to MLP_Block
        return x
# ----------------------------------------------------------------------------------------------------------------------


class SSFTT_Net(nn.Module):

    """
    https://ieeexplore.ieee.org/document/9684381
    """

    def __init__(self,
                 in_channels=1,
                 n_classes=3,
                 num_tokens=4,
                 n_bands=28,
                 dim=64,
                 depth=1,
                 heads=8,
                 mlp_dim=8,
                 dropout=0.1,
                 emb_dropout=0.1):
        super(SSFTT_Net, self).__init__()
        self.L = num_tokens
        self.cT = dim
        self.conv3d_features = nn.Sequential(nn.Conv3d(in_channels,
                                                       out_channels=8,
                                                       kernel_size=(3, 3, 3),
                                                       bias=False),
                                             nn.BatchNorm3d(8),
                                             nn.ReLU())

        self.conv2d_features = nn.Sequential(nn.Conv2d(in_channels=8*(n_bands-2),
                                                       out_channels=64,
                                                       kernel_size=(3, 3),
                                                       bias=False),
                                             nn.BatchNorm2d(64),
                                             nn.ReLU())

        # Tokenization
        self.token_wA = nn.Parameter(torch.empty(1, self.L, 64),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wA)
        self.token_wV = nn.Parameter(torch.empty(1, 64, self.cT),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wV)

        self.pos_embedding = nn.Parameter(torch.empty(1, (num_tokens + 1), dim))
        torch.nn.init.normal_(self.pos_embedding, std=.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)

        self.to_cls_token = nn.Identity()

        self.nn1 = nn.Linear(dim, n_classes)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std=1e-6)

    def forward(self, x, mask=None):

        x = self.conv3d_features(x)
        x = rearrange(x, 'b c h w y -> b (c h) w y')
        x = self.conv2d_features(x)
        x = rearrange(x, 'b c h w -> b (h w) c')

        wa = rearrange(self.token_wA, 'b h w -> b w h')  # Transpose
        A = torch.einsum('bij,bjk->bik', x, wa)
        A = rearrange(A, 'b h w -> b w h')  # Transpose
        A = A.softmax(dim=-1)

        VV = torch.einsum('bij,bjk->bik', x, self.token_wV)
        T = torch.einsum('bij,bjk->bik', A, VV)

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, T), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        x = self.transformer(x, mask)  # main game
        x = self.to_cls_token(x[:, 0])
        x = self.nn1(x)

        return x
# ----------------------------------------------------------------------------------------------------------------------


class SSFTT(Model):
    def __init__(self,
                 n_classes,
                 device,
                 n_bands,
                 path_to_weights=None
                 ):
        super(SSFTT, self).__init__()
        self.hyperparams: dict[str: Any] = dict()
        self.hyperparams['patch_size'] = 13
        self.hyperparams['n_classes'] = n_classes
        self.hyperparams['ignored_labels'] = [0]
        self.hyperparams['device'] = device
        self.hyperparams['n_bands'] = n_bands
        self.hyperparams['center_pixel'] = True
        self.hyperparams['net_name'] = 'ssftt'

        self.model = SSFTT_Net(in_channels=1, n_bands=n_bands, n_classes=n_classes)

        if path_to_weights:
            self.model.load_state_dict(torch.load(path_to_weights))

        self.hyperparams.setdefault("supervision", "full")
        self.hyperparams.setdefault("flip_augmentation", False)
        self.hyperparams.setdefault("radiation_augmentation", False)
        self.hyperparams.setdefault("mixture_augmentation", False)
        self.hyperparams["center_pixel"] = True

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            fit_params: Dict):
        fit_params.setdefault('batch_size', 32)
        self.hyperparams['batch_size'] = fit_params['batch_size']

        img, gt = get_dataset(hsi=X, mask=y)

        train_gt, _ = sample_gt(gt=gt,
                                train_size=fit_params['train_sample_percentage'],
                                mode=fit_params['dataloader_mode'],
                                msg='train_val/test')

        train_gt, val_gt = sample_gt(gt=train_gt,
                                     train_size=0.9,
                                     mode=fit_params['dataloader_mode'],
                                     msg='train/val')
        self.train_mask = train_gt
        train_loader = create_torch_loader(img, train_gt, self.hyperparams, shuffle=True)
        val_loader = create_torch_loader(img, val_gt, self.hyperparams)

        fit_params.setdefault('epochs', 10)
        fit_params.setdefault('train_sample_percentage', 0.5)
        fit_params.setdefault('dataloader_mode', 'random')
        fit_params.setdefault('loss', nn.CrossEntropyLoss())
        fit_params.setdefault('optimizer_params', {'learning_rate': 0.001, 'weight_decay': 0})
        fit_params.setdefault('optimizer',
                              optim.Adam(self.model.parameters(),
                                         lr=fit_params['optimizer_params']["learning_rate"],
                                         weight_decay=fit_params['optimizer_params']['weight_decay']))
        fit_params.setdefault('scheduler_type', None)
        fit_params.setdefault('scheduler_params', None)
        fit_params.setdefault('wandb', self.wandb_run)
        fit_params.setdefault('tensorboard', self.writer)

        if fit_params['scheduler_type'] == 'StepLR':
            scheduler = optim.lr_scheduler.StepLR(optimizer=fit_params['optimizer'],
                                                  **fit_params['scheduler_params'])
        elif fit_params['scheduler_type'] == 'CosineAnnealingLR':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=fit_params['optimizer'],
                                                             **fit_params['scheduler_params'])
        elif fit_params['scheduler_type'] == 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=fit_params['optimizer'],
                                                             **fit_params['scheduler_params'])
        elif fit_params['scheduler_type'] is None:
            scheduler = None
        else:
            raise ValueError('Unsupported scheduler type')

        self.model, history = train(net=self.model,
                                    optimizer=fit_params['optimizer'],
                                    criterion=fit_params['loss'],
                                    scheduler=scheduler,
                                    epoch=fit_params['epochs'],
                                    data_loader=train_loader,
                                    val_loader=val_loader,
                                    device='cuda',
                                    wandb_run=fit_params['wandb'],
                                    writer=fit_params['tensorboard']
                                    )
        save_train_mask(model_name=camel_to_snake(str(self.model.__class__.__name__)),
                        dataset_name=train_loader.dataset.name,
                        mask=train_gt)

        self.train_loss = history["train_loss"]
        self.val_loss = history["val_loss"]
        self.train_accs = history["train_accuracy"]
        self.val_accs = history["val_accuracy"]
        self.lrs = history["lr"]
    # ------------------------------------------------------------------------------------------------------------------

    def predict(self,
                X: np.ndarray,
                y: Optional[np.ndarray] = None,
                batch_size=32) -> np.ndarray:

        self.hyperparams["test_stride"] = 1
        self.hyperparams["batch_size"] = batch_size

        prediction = super().predict_nn(X=X,
                                        y=y,
                                        model=self.model,
                                        hyperparams=self.hyperparams)

        return prediction
# ----------------------------------------------------------------------------------------------------------------------
