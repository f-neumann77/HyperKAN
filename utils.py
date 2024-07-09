import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def get_palette(num_classes):
    palette = {0: (0, 0, 0)}
    for k, color in enumerate(sns.color_palette("hls", num_classes)):
        palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype="uint8"))

    return palette
# ----------------------------------------------------------------------------------------------------------------------


def convert_to_color(arr_2d, palette=None):
    """Convert an array of labels to RGB color-encoded image.

    Args:
        arr_2d: int 2D array of labels
        palette: dict of colors used (label number -> RGB tuple)

    Returns:
        arr_3d: int 2D images of color-encoded labels in RGB format

    """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    if palette is None:
        palette = get_palette(np.max(arr_2d))

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d
# ----------------------------------------------------------------------------------------------------------------------


def draw_colored_mask(mask: np.ndarray,
                      predicted_mask: np.array = None,
                      mask_labels: dict = None,
                      stack_type: str = 'v'):
    palette = get_palette(np.max(mask))

    color_gt = convert_to_color(mask, palette=palette)

    t = 1
    tmp = lambda x: [i / 255 for i in x]
    cmap = {k: tmp(rgb) + [t] for k, rgb in palette.items()}

    # patches = [mpatches.Patch(color=cmap[i], label=mask_labels.get(str(i), 'no information')) for i in cmap]

    plt.figure(figsize=(12, 12))
    if np.any(predicted_mask):
        color_pred = convert_to_color(predicted_mask, palette=palette)
        if stack_type == 'v':
            combined = np.vstack((color_gt, color_pred))
        elif stack_type == 'h':
            combined = np.hstack((color_gt, color_pred))
        else:
            raise Exception(f'{stack_type} is unresolved mode')
        plt.imshow(combined, label='Colored ground truth and predicted masks')
    else:
        plt.imshow(color_gt, label='Colored ground truth mask')
    # if labels:
    #    plt.legend(handles=patches, loc=4, borderaxespad=0.)
    plt.show()

    return color_gt
