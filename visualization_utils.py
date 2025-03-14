# This file is borrowed from https://github.com/RomeroBarata/human_object_interaction/blob/main/vhoi/visualisation.py

import datetime
import os
from itertools import groupby
from typing import Dict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def create_label_bar(label_ids: list, bar_height: int = 30, bar_width: int = 5):
    """Create a numpy array that represents the video segmentation.

    Arguments:
        label_ids - Video segmentation represented as a list of label IDs.
        bar_height - Height (or number of rows) of the returned numpy array.
        bar_width - Width of individual segments in the desired plot. The returned array has len(label_ids) * bar_width
            columns.
    Returns:
        A numpy array of shape (bar_height, len(label_ids) * bar_width) containing a representation of video
        segmentation.
    """
    label_bar = np.empty([bar_height, bar_width * len(label_ids)])
    for i, label in enumerate(label_ids):
        label_bar[:, i * bar_width:(i + 1) * bar_width] = label
    return label_bar


def determine_xlabels_and_xticks_positions(labels: list, bar_width: int):
    """Simplify segmentation labelling in case of frame-wise segmentation.

    From a list of frame-level labels, extract the unique labels and determine x-axis positions to plot them.

    Arguments:
        labels - Video segmentation as a list of labels.
        bar_width - Width of a single segment bar in the expected plot.
    Returns:
        Two lists. The first one contains the unique labels in labels, and the second contain the x-axis position to
        place the labels in the final segmentation plot.
    """
    unique_labels, xticks, cumulative_length = [], [], 0
    for k, v in groupby(labels):
        unique_labels.append(k)
        num_frames = len(list(v))
        if xticks:
            xticks.append(cumulative_length + (num_frames // 3))
        else:
            xticks.append(num_frames // 3)
        xticks[-1] *= bar_width
        cumulative_length += num_frames
    return unique_labels, xticks


def plot_segmentation(target: list, *output, class_id_to_label: Dict[int, str], save_file: str = None,
                      bar_height: int = 30, bar_width: int = 2000, xlabels_type: str = 'label'):
    """Plot ground-truth and predicted segmentations.

    Arguments:
        target - A list containing the ground-truth label IDs.
        output - Output predictions to compare against the target. Each element is a list containing the predicted
            labels IDs.
        class_id_to_label - Dictionary mapping label IDs to label names.
        save_file - Optional file to write out segmentation plot.
        bar_height - Height of the bars drawn.
        bar_width - Width of the bars drawn.
        xlabels_type - One of 'label', 'id', or None.
    """
    bar_width = int(bar_width / len(target))
    num_classes = len(class_id_to_label)
    plt.figure(figsize=(num_classes, 1))
    grid_spec = mpl.gridspec.GridSpec(1 + len(output), 1)
    grid_spec.update(wspace=0.5, hspace=0.01)
    for plt_idx, label_ids in enumerate([target, *output]):
        ax = plt.subplot(grid_spec[plt_idx])
        label_bar = create_label_bar(label_ids, bar_height=bar_height, bar_width=bar_width)
        plt.imshow(label_bar, cmap=plt.get_cmap('tab20'), vmin=0, vmax=num_classes - 1)
        ax.tick_params(axis='both', which='both', length=0)
        xlabels, xticks = determine_xlabels_and_xticks_positions(label_ids, bar_width)
        ax.set_xticks(xticks)
        fontsize = 'small'
        if xlabels_type == 'labels':
            xlabels, fontsize = [class_id_to_label[label_id] for label_id in xlabels], 'x-small'
        elif xlabels_type == 'id':
            xlabels = [str(label_id) for label_id in xlabels]
        else:
            xlabels = []
        ax.set_xticklabels(xlabels, fontsize=fontsize, horizontalalignment='left')
        ax.set_yticklabels([])
    if save_file is not None:
        plt.savefig(save_file, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()
    plt.close()


def plot_segmentation_v2(y_true, y_pred, conf, y_pred_M, conf_M, save_file, title):
    # This is a very custom function to plot the segmentation of the video.

    bar_width = 1
    bar_height = 300

    # y_true = np.repeat(np.repeat(y_true.reshape(1, -1), bar_height, axis=0), bar_width, axis=1)
    # y_pred = np.repeat(np.repeat(y_pred.reshape(1, -1), bar_height, axis=0), bar_width, axis=1)
    # y_pred_M = np.repeat(np.repeat(y_pred_M.reshape(1, -1), bar_height, axis=0), bar_width, axis=1)

    y_true = create_label_bar(y_true, bar_height=bar_height, bar_width=bar_width).astype(np.uint8)
    y_pred = create_label_bar(y_pred, bar_height=bar_height, bar_width=bar_width).astype(np.uint8)
    y_pred_M = create_label_bar(y_pred_M, bar_height=bar_height, bar_width=bar_width).astype(np.uint8)

    # Plot the segmentation
    # plt.figure(figsize=(30, 10))
    fig, axs = plt.subplots(5, 1, sharex=True, figsize=(30, 10))
    plt.suptitle(title)

    # Plot the ground truth
    axs[0].set_title(f'Ground Truth: {title}')
    axs[0].imshow(y_true, cmap=plt.get_cmap('tab20'), vmin=0, vmax=20, aspect='auto')
    
    axs[1].set_title('Pred')
    axs[1].imshow(y_pred, cmap=plt.get_cmap('tab20'), vmin=0, vmax=20, aspect='auto')
    
    axs[2].set_title('Confidence')
    axs[2].plot(conf, color='black', linewidth=0.7)
    
    axs[3].set_title('Pred (M)')
    axs[3].imshow(y_pred_M, cmap=plt.get_cmap('tab20'), vmin=0, vmax=20, aspect='auto')
    
    axs[4].set_title('Confidence')
    axs[4].plot(conf_M, color='black', linewidth=0.7)
    
    
    fig.tight_layout()
    fig.savefig(save_file)
    plt.close()

def plot_segmentation_v3(y_true_L, y_pred_L, conf_L, y_true_R, y_pred_R, conf_R, save_file, title):
    # This is a very custom function to plot the segmentation of the video.

    bar_width = 1
    bar_height = 300

    # y_true = np.repeat(np.repeat(y_true.reshape(1, -1), bar_height, axis=0), bar_width, axis=1)
    # y_pred = np.repeat(np.repeat(y_pred.reshape(1, -1), bar_height, axis=0), bar_width, axis=1)
    # y_pred_M = np.repeat(np.repeat(y_pred_M.reshape(1, -1), bar_height, axis=0), bar_width, axis=1)

    y_true_L = create_label_bar(y_true_L, bar_height=bar_height, bar_width=bar_width).astype(np.uint8)
    y_pred_L = create_label_bar(y_pred_L, bar_height=bar_height, bar_width=bar_width).astype(np.uint8)
    y_true_R = create_label_bar(y_true_R, bar_height=bar_height, bar_width=bar_width).astype(np.uint8)
    y_pred_R = create_label_bar(y_pred_R, bar_height=bar_height, bar_width=bar_width).astype(np.uint8)


    # Plot the segmentation
    # plt.figure(figsize=(30, 10))
    fig, axs = plt.subplots(6, 1, sharex=True, figsize=(30, 10))
    plt.suptitle(title)

    # Plot the ground truth
    axs[0].set_title(f'Left Hand: Ground Truth')
    axs[0].imshow(y_true_L, cmap=plt.get_cmap('tab20'), vmin=0, vmax=20, aspect='auto')
    
    axs[1].set_title('Left Hand: Prediction')
    axs[1].imshow(y_pred_L, cmap=plt.get_cmap('tab20'), vmin=0, vmax=20, aspect='auto')
    
    axs[2].set_title('Left Hand: Confidence')
    axs[2].plot(conf_L, color='black', linewidth=0.7)
    
    axs[3].set_title('Right Hand: Ground Truth')
    axs[3].imshow(y_true_R, cmap=plt.get_cmap('tab20'), vmin=0, vmax=20, aspect='auto')
    
    axs[4].set_title('Right Hand: Prediction')
    axs[4].imshow(y_pred_R, cmap=plt.get_cmap('tab20'), vmin=0, vmax=20, aspect='auto')
    
    axs[5].set_title('Right Hand: Confidence')
    axs[5].plot(conf_R, color='black', linewidth=0.7)

    
    fig.tight_layout()
    fig.savefig(save_file)
    plt.close()


def plot_segmentation_partial(y_pred_dict, y_true, save_dir, title):
    # This is a very custom function to plot the partial segments, and save it the temp folder.
    # This function will be called from the training loop.

    # y_pred_dict: [epoch]: [y_true, y_pred]

    save_dir = os.path.join(save_dir, "partial_segments")
    os.makedirs(save_dir, exist_ok=True)

    bar_width = 1
    bar_height = 150

    # Plot the segmentation
    # num_col = 3
    num_epochs = len(y_pred_dict)
    # num_row = (num_epochs+num_col-1) // num_col

    # fig, axs = plt.subplots(num_row, num_col, sharex=True, figsize=(30, 10))
    fig, axs = plt.subplots(num_epochs+1, 1, sharex=True, figsize=(8, (num_epochs+1)*3))
    plt.suptitle(title)

    # Plot the ground truth
    y_true = create_label_bar(y_true, bar_height=bar_height, bar_width=bar_width).astype(np.uint8)
    axs[0].set_title(f'Ground Truth: {title}')
    axs[0].imshow(y_true, cmap=plt.get_cmap('tab20'), vmin=0, vmax=20, aspect='auto')
    
    # epoch starts from 1
    for e, y_pred in y_pred_dict.items():
        # x_pos = e % num_col
        # y_pos = e // num_col
        # axs[x_pos, y_pos].set_title(f'P{e}')
        axs[e].set_title(f'P{e}')

        y_pred = create_label_bar(y_pred, bar_height=bar_height, bar_width=bar_width).astype(np.uint8)
        # axs[x_pos, y_pos].imshow(y_pred, cmap=plt.get_cmap('tab20'), vmin=0, vmax=20, aspect='auto')
        axs[e].imshow(y_pred, cmap=plt.get_cmap('tab20'), vmin=0, vmax=20, aspect='auto')
    
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"{title}.png"))
    plt.close()