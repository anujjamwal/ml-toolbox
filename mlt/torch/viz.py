import torch
import matplotlib.pyplot as plt
import numpy as np


def show_conv2d_rgb(layer: torch.nn.Conv2d, fig_num_cols=10):
    weights = layer.weight.data.cpu()
    num_kernels = weights.shape[0]

    fig = plt.figure(figsize=(fig_num_cols, num_kernels))

    for i in range(num_kernels):
        ax = fig.add_subplot(num_kernels, fig_num_cols, i + 1)
        npimg = np.array(weights[i].numpy(), np.float32)
        npimg = (npimg - np.mean(npimg)) / np.std(npimg)
        npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
        npimg = npimg.transpose((1, 2, 0))
        ax.imshow(npimg)
        ax.axis('off')
        ax.set_title(str(i))
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    plt.tight_layout()
    plt.show()


def show_conv2d(layer: torch.nn.Conv2d, fig_num_cols=10):
    weights = layer.weight.data.cpu()
    num_kernels = weights.shape[0] * weights.shape[1]

    nrows = 1 + num_kernels//fig_num_cols
    count = 0
    fig = plt.figure(figsize=(fig_num_cols, nrows))

    # looping through all the kernels in each channel
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            count += 1
            ax1 = fig.add_subplot(nrows, fig_num_cols, count)
            npimg = np.array(weights[i, j].numpy(), np.float32)
            npimg = (npimg - np.mean(npimg)) / np.std(npimg)
            npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
            ax1.imshow(npimg)
            ax1.set_title(str(i) + ',' + str(j))
            ax1.axis('off')
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])

    plt.tight_layout()
    plt.show()
