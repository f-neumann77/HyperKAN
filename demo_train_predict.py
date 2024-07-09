import numpy as np
from scipy.io import loadmat

# Import neural networks models
from models import MLP, MLP_KAN, Hu1DCNN, Hu1DCNN_KAN, \
                   M1DCNN, M1DCNN_KAN, Luo3DCNN, Luo3DCNN_KAN, \
                   He3DCNN, He3DCNN_KAN, NM3DCNN, NM3DCNN_KAN, \
                   SSFTT, SSFTT_KAN

# Import helper tools
from models.utils import draw_fit_plots
from dataloaders.utils import HyperStandardScaler
from datasets_config import PaviaU, PaviaC, Salinas, IP, H13, H18, KSC

from models.utils import get_accuracy, get_f1
from utils import draw_colored_mask

# Import pca wrapper for hsi
from dataloaders.utils import apply_pca

DATASET = H18
NN_MODEL = SSFTT_KAN

optimizer_params = {
    "learning_rate": 0.001,
    "weight_decay": 0
}

scheduler_params = {
    "step_size": 5,
    "gamma": 0.5
}

augmentation_params = {
    "flip_augmentation": False,
    "radiation_augmentation": False,
    "mixture_augmentation": False
}

fit_params = {
    "epochs": 25,
    "train_sample_percentage": DATASET.get('train_sample_percentage'),
    "dataloader_mode": "fixed",
    "wandb_vis": False,
    "optimizer_params": optimizer_params,
    "batch_size": 256,
    "scheduler_type": 'StepLR',
    "scheduler_params": scheduler_params
}


def predict_(X,
             y,
             y_train=None,
             cnn=None):

    pred = cnn.predict(X=X,
                       y=y,
                       batch_size=512)

    pred = pred * (y > 0)
    mask_ = y * (y_train == 0)
    pred = pred * (y_train == 0)
    draw_colored_mask(mask=y,
                      predicted_mask=pred,
                      stack_type='h')
    return get_accuracy(target=mask_, prediction=pred), get_f1(target=mask_, prediction=pred, average='weighted')


hsi_path = DATASET.get('path_to_hsi')
hsi_key = DATASET.get('hsi_key')
mask_path = DATASET.get('path_to_mask')
mask_key = DATASET.get('mask_key')

hsi = loadmat(hsi_path)[hsi_key]
mask = loadmat(mask_path)[mask_key]

n_classes = len(np.unique(mask))

scaler = HyperStandardScaler()
hsi = scaler.fit_transform(hsi)

hsi, pca = apply_pca(hsi, 30)

cnn = NN_MODEL(n_classes=n_classes,
               n_bands=hsi.shape[-1],
               device='cuda')

cnn.fit(X=hsi,  # or hsi
        y=mask,
        fit_params=fit_params)

draw_fit_plots(model=cnn)

acc_bl, f1_bl = predict_(hsi, mask, cnn=cnn, y_train=cnn.train_mask)

print(acc_bl, f1_bl)
