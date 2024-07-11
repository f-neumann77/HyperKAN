import copy
import datetime
import numpy as np
import os
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as udata

from abc import ABC, abstractmethod
from torcheval.metrics import MulticlassAccuracy
from tqdm import trange, tqdm
from typing import Dict, Iterable, Literal

from dataloaders.torch_dataloader import create_torch_loader
from dataloaders.utils import get_dataset, grouper, count_sliding_window, sliding_window, sample_gt
from models.utils import camel_to_snake, EarlyStopping


SchedulerTypes = Literal['StepLR', 'CosineAnnealingLR', 'ReduceLROnPlateau']
OptimizerTypes = Literal['SGD', 'Adam', 'Adagrad']


class Model(ABC):
    """
    Model()

        Abstract class for decorating machine learning algorithms

    """

    @abstractmethod
    def __init__(self):
        self.train_loss = []
        self.val_loss = []
        self.train_accs = []
        self.val_accs = []
        self.lrs = []
        self.model = None
        self.wandb_run = None
        self.writer = None
        self.train_mask = None

    @abstractmethod
    def fit(self,
            X,
            y,
            fit_params):
        raise NotImplemented("Method fit must be implemented!")
    # ------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def predict(self,
                X,
                y) -> np.ndarray:
        raise NotImplemented("Method predict must be implemented!")
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def fit_nn(X,
               y,
               hyperparams,
               model,
               fit_params):
        """

        Parameters
        ----------
        X
        y
        hyperparams
        model
        fit_params

        Returns
        -------

        """
        img, gt = get_dataset(hsi=X, mask=y)

        hyperparams['batch_size'] = fit_params['batch_size']

        scheduler = get_scheduler(scheduler_type=fit_params['scheduler_type'],
                                  optimizer=fit_params['optimizer'],
                                  scheduler_params=fit_params['scheduler_params'])

        train_gt, _ = sample_gt(gt=gt,
                                train_size=fit_params['train_sample_percentage'],
                                mode=fit_params['dataloader_mode'],
                                msg='train_val/test')

        train_gt, val_gt = sample_gt(gt=train_gt,
                                     train_size=0.9,
                                     mode=fit_params['dataloader_mode'],
                                     msg='train/val')

        print(f'Full size: {np.sum(gt > 0)}')
        print(f'Train size: {np.sum(train_gt > 0)}')
        print(f'Val size: {np.sum(val_gt > 0)}')

        # Generate the dataset
        train_loader = create_torch_loader(img, train_gt, hyperparams, shuffle=True)
        val_loader = create_torch_loader(img, val_gt, hyperparams)

        save_train_mask(model_name=camel_to_snake(str(model.__class__.__name__)),
                        dataset_name=train_loader.dataset.name,
                        mask=train_gt)

        model, history = train(net=model,
                               optimizer=fit_params['optimizer'],
                               criterion=fit_params['loss'],
                               scheduler=scheduler,
                               data_loader=train_loader,
                               epoch=fit_params['epochs'],
                               val_loader=val_loader,
                               device=hyperparams['device'])

        return model, history, train_gt
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def predict_nn(model,
                   X,
                   y=None,
                   hyperparams=None):
        hyperparams["test_stride"] = 1
        hyperparams.setdefault('batch_size', 1)
        img, gt = get_dataset(X, mask=None)

        model.eval()

        probabilities = test(net=model,
                             img=img,
                             hyperparams=hyperparams)
        prediction = np.argmax(probabilities, axis=-1)
        # fill void areas in result with zeros
        if y is not None:
            prediction[y == 0] = 0
        return prediction
    # ------------------------------------------------------------------------------------------------------------------


def get_optimizer(net: nn.Module,
                  optimizer_type: OptimizerTypes,
                  optimizer_params: Dict):
    if optimizer_type == 'SGD':
        optimizer = optim.SGD(net.parameters(),
                              **optimizer_params)
    elif optimizer_type == 'Adam':
        optimizer = optim.Adam(net.parameters(),
                               **optimizer_params)
    elif optimizer_type == 'Adagrad':
        optimizer = optim.Adagrad(net.parameters(),
                                  **optimizer_params)
    else:
        raise ValueError('Unsupported optimizer type')

    return optimizer
# ----------------------------------------------------------------------------------------------------------------------


def get_scheduler(scheduler_type: SchedulerTypes,
                  optimizer: optim.lr_scheduler,
                  scheduler_params: dict):

    if scheduler_type == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer,
                                              **scheduler_params)
    elif scheduler_type == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                         **scheduler_params)
    elif scheduler_type == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                         **scheduler_params)
    elif scheduler_type is None:
        scheduler = None
    else:
        raise ValueError('Unsupported scheduler type')

    return scheduler
# ----------------------------------------------------------------------------------------------------------------------


def __calc_acc(output,
               target,
               device):
    metric = MulticlassAccuracy(device=device)
    _, output = torch.max(output, dim=1)

    indices = torch.nonzero(output)
    output = output[indices]
    target = target[indices]
    metric.update(output.view(-1), target.view(-1))
    acc = metric.compute()

    return acc
# ----------------------------------------------------------------------------------------------------------------------


def train_one_epoch(net: nn.Module,
                    criterion: nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device):
    net.train()
    avg_train_loss = 0.0
    train_accuracy = []
    # Run the training loop for one epoch
    losses = []
    for batch_idx, (data, target) in (enumerate(data_loader)):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        output = net(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        avg_train_loss += loss.item()
        losses.append(loss.item())

        acc = __calc_acc(output=output,
                         target=target,
                         device=device)

        train_accuracy.append(acc.to('cpu'))

    train_metrics = dict()

    train_metrics["avg_train_loss"] = avg_train_loss / len(data_loader)
    train_metrics["train_acc"] = np.mean(train_accuracy)
    train_metrics["current_lr"] = optimizer.param_groups[0]['lr']

    return train_metrics
# ----------------------------------------------------------------------------------------------------------------------


def val_one_epoch(net: nn.Module,
                  criterion: nn.Module,
                  data_loader: udata.DataLoader,
                  device: torch.device):
    """

    Parameters
    ----------
    net: nn.Module
        neural network model
    criterion: nn.Module
    data_loader
    device

    Returns
    -------

    """

    val_accs = []
    avg_loss = 0

    for batch_idx, (data, target) in enumerate(data_loader):
        with torch.no_grad():
            # Load the data into the GPU if required
            data, target = data.to(device), target.to(device)
            output = net(data)
            loss = criterion(output, target)
            avg_loss += loss.item()
            acc = __calc_acc(output=output,
                             target=target,
                             device=device)

            val_accs.append(acc.to('cpu'))

    return np.mean(val_accs), avg_loss / len(data_loader)
# ----------------------------------------------------------------------------------------------------------------------


def train(net: nn.Module,
          optimizer: torch.optim,
          criterion,
          scheduler: torch.optim.lr_scheduler,
          data_loader: udata.DataLoader,
          epoch,
          device=None,
          val_loader=None
          ):
    """
    Training loop to optimize a network for several epochs and a specified loss
    Parameters
    ----------
        net:
            a PyTorch model
        optimizer:
            a PyTorch optimizer
        criterion:
            a PyTorch-compatible loss function, e.g. nn.CrossEntropyLoss
        scheduler:
            PyTorch scheduler
        data_loader:
            a PyTorch dataset loader
        epoch:
            int specifying the number of training epochs
        device:
            torch device to use (defaults to CPU)
        val_loader:
            validation dataset
    """
    net.to(device)

    save_epoch = epoch // 20 if epoch > 20 else 1

    early_stopping = EarlyStopping(tolerance=5, min_delta=10)

    train_accuracies = []
    val_accuracies = []
    train_loss = []
    val_loss = []
    lrs = []
    t = trange(1, epoch + 1, desc='Train loop', leave=True, dynamic_ncols=True)
    best_weights = None
    val_acc_best = 0
    for e in t:
        train_metrics = train_one_epoch(net=net,
                                        criterion=criterion,
                                        data_loader=data_loader,
                                        optimizer=optimizer,
                                        device=device)

        train_accuracies.append(train_metrics["train_acc"])
        train_loss.append(train_metrics["avg_train_loss"])
        lrs.append(train_metrics["current_lr"])
        val_metrics = dict()

        if val_loader:
            val_metrics['val_acc'], val_metrics['avg_val_loss'] = val_one_epoch(net,
                                                                                criterion,
                                                                                val_loader,
                                                                                device=device)

            t.set_postfix({'train_acc': "{:.3f}".format(train_metrics["train_acc"]),
                           'val_acc': "{:.3f}".format(val_metrics['val_acc']),
                           'train_loss': "{:.3f}".format(train_metrics["avg_train_loss"]),
                           'val_loss': "{:.3f}".format(val_metrics['avg_val_loss']),
                           'lr': optimizer.param_groups[0]['lr']
                           }
                          )

            val_loss.append(val_metrics['avg_val_loss'])
            val_accuracies.append(val_metrics['val_acc'])
            metric = val_metrics['val_acc']
            if val_metrics['val_acc'] > val_acc_best:
                best_weights = copy.deepcopy(net)
        else:
            metric = train_metrics["avg_train_loss"]

        # Update the scheduler
        if scheduler is not None:
            scheduler.step()

        # Save the weights
        #if e % save_epoch == 0:
        #    save_model(
        #       net,
        #       camel_to_snake(str(net.__class__.__name__)),
        #       data_loader.dataset.name,
        #       epoch=e,
        #       metric=abs(metric),
         #   )

        # Early stopping
        early_stopping(train_metrics["avg_train_loss"], val_metrics['avg_val_loss'])
        if early_stopping.early_stop:
            print("Early stopping")
            break

    save_model(
               net,
               camel_to_snake(str(net.__class__.__name__)),
               data_loader.dataset.name,
               epoch='best_val_acc',
               metric=max(val_accuracies))

    history = dict()
    history["train_loss"] = train_loss
    history["val_loss"] = val_loss
    history["train_accuracy"] = train_accuracies
    history["val_accuracy"] = val_accuracies
    history["lr"] = lrs

    df = pd.DataFrame(history)
    df.to_csv('metrics.csv')

    return best_weights, history
# ----------------------------------------------------------------------------------------------------------------------


def test(net: nn.Module,
         img: np.ndarray,
         hyperparams):
    """
    Test a model on a specific image
    """

    patch_size = hyperparams["patch_size"]
    center_pixel = hyperparams["center_pixel"]
    batch_size, device = hyperparams["batch_size"], hyperparams["device"]
    n_classes = hyperparams["n_classes"]

    kwargs = {
        "step": hyperparams["test_stride"],
        "window_size": (patch_size, patch_size),
    }
    probs = np.zeros(img.shape[:2] + (n_classes,))
    net.to(device)
    iterations = count_sliding_window(img, **kwargs) // batch_size

    t = tqdm(grouper(batch_size, sliding_window(img, **kwargs)),
             total=iterations,
             desc="Inference on the image")

    for batch in t:
        with torch.no_grad():
            if patch_size == 1:
                data = [b[0][0, 0] for b in batch]
                data = np.copy(data)
                data = torch.from_numpy(data)
            else:
                data = [b[0] for b in batch]
                data = np.copy(data)
                data = data.transpose(0, 3, 1, 2)
                data = torch.from_numpy(data)
                data = data.unsqueeze(1)

            indices = [b[1:] for b in batch]
            data = data.to(device, dtype=torch.float)
            output = net(data)

            if isinstance(output, tuple):
                output = output[0]
            output = output.to("cpu")

            output = output.numpy()
            if patch_size != 1 and not center_pixel:
                output = np.transpose(output, (0, 2, 3, 1))

            for (x, y, w, h), out in zip(indices, output):
                if center_pixel:
                    probs[x + w // 2, y + h // 2] += out
                else:
                    probs[x: x + w, y: y + h] += out

    return probs
# ----------------------------------------------------------------------------------------------------------------------


def save_train_mask(model_name, dataset_name, mask):

    mask_dir = "./masks/" + model_name + "/" + dataset_name + "/"
    time_str = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    if not os.path.isdir(mask_dir):
        os.makedirs(mask_dir, exist_ok=True)
    gray_filename = f"{mask_dir}/{time_str}_gray_mask.npy"

    # TODO check what's wrong here
    color_filename = f"{mask_dir}/{time_str}_color_mask.png"

    np.save(gray_filename, mask)
# ----------------------------------------------------------------------------------------------------------------------


def save_model(model,
               model_name,
               dataset_name,
               **kwargs):
    model_dir = "./checkpoints/" + model_name + "/" + dataset_name + "/"
    """
    Using strftime in case it triggers exceptions on windows 10 system
    """
    time_str = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    if isinstance(model, torch.nn.Module):
        filename = time_str + "_epoch{epoch}_{metric:.2f}".format(
            **kwargs
        )
        torch.save(model.state_dict(), model_dir + filename + ".pth")
    else:
        print('Saving error')
