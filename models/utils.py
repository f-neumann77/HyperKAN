import numpy as np
import os
import re
import wandb
import yaml

from collections import OrderedDict
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.tensorboard import SummaryWriter
from typing import List, Union


def camel_to_snake(name):
    s = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s).lower()
# ----------------------------------------------------------------------------------------------------------------------


class EarlyStopping:
    """
        EarlyStopping class

        Attributes
        ----------
        tolerance: int
            number of epochs to wait after min has been hit
        min_delta: float
            minimum change in the monitored quantity to qualify as an improvement
        counter: int
            number of epochs since min has been hit
        early_stop: bool
            True if the training process has to be stopped

        Methods
        -------
        __call__(train_loss, validation_loss)
            call method to check if the training process has to be stopped
    """

    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True
# ----------------------------------------------------------------------------------------------------------------------


def draw_fit_plots(model):
    """
    draw_fit_plots(model)

        Draws plot of train/val loss and plot of train/val accuracy after model fitting

        Parameters
        ----------
        model:
            model of neural network

    """
    x = [int(i) for i in range(1, len(model.train_loss) + 1)]

    plt.figure(figsize=(12, 8))
    plt.plot(x, model.train_loss, c='green', label="train loss")
    plt.plot(x, model.val_loss, c='blue', label="validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.xticks(x)
    plt.grid()
    plt.legend()
    plt.savefig('TrainVal_losses_plot.png')
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.plot(x, model.train_accs, c='green', label='train accuracy')
    plt.plot(x, model.val_accs, c='blue', label="validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.xticks(x)
    plt.grid()
    plt.legend()
    plt.savefig('TrainVal_accs.png')
    plt.show()

    if model.lrs:
        plt.figure(figsize=(12, 8))
        plt.plot(x, model.lrs, c='blue', label='Learning rate')
        plt.xlabel("Epochs")
        plt.ylabel("Learning rate")
        plt.xticks(x)
        plt.grid()
        plt.legend()
        plt.savefig('Learning_rate.png')
        plt.show()
# ----------------------------------------------------------------------------------------------------------------------


def __prepare_pred_target(prediction: np.ndarray,
                          target: np.ndarray):
    """
    Remove all zeros masked pixels from prediction and target

    Parameters
    ----------
    prediction
    target

    Returns
    -------

    """
    prediction = prediction.flatten()

    target = target.flatten()

    # remove all pixels with zero-value mask
    indices = np.nonzero(target*prediction)
    prediction = prediction[indices]
    target = target[indices]

    return prediction, target
# ----------------------------------------------------------------------------------------------------------------------


def get_accuracy(prediction: np.ndarray,
                 target: np.ndarray,
                 *args,
                 **kwargs):
    prediction, target = __prepare_pred_target(prediction, target)

    return accuracy_score(y_true=target, y_pred=prediction, *args, **kwargs)
# ----------------------------------------------------------------------------------------------------------------------


def get_f1(prediction: np.ndarray,
           target: np.ndarray,
           *args,
           **kwargs):
    prediction, target = __prepare_pred_target(prediction, target)

    return f1_score(y_true=target, y_pred=prediction, *args, **kwargs)
# ----------------------------------------------------------------------------------------------------------------------


def get_confusion_matrix():
    # TODO realise it
    pass
# ----------------------------------------------------------------------------------------------------------------------
