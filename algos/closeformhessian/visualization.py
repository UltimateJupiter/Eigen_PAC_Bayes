import os, sys
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import cm
from matplotlib import gridspec
import matplotlib.pyplot as plt
from .utils import *


def get_image_path(name):
    vis_dir = "/usr/xtmp/CSPlus/VOLDNN/Shared/Visualizations/"
    img_path = os.path.join(vis_dir, "{}.jpg".format(name))
    print("https://users.cs.duke.edu/~xz231" + img_path.split("Shared")[1] + '\n')
    return img_path

class vis():

    def __init__(self, device):
        self.device = device
        return
    
    def plots(self, xs, ys, labels, name='tmp',
        x_label=None, y_label=None,
        s=6, fig_size=(8, 6), dpi=100, line_style='-', marker_style=None,
        x_log=False, y_log=False):
        log("Plotting {}".format(name))
        plt.figure(figsize=fig_size)
        assert len(xs) == len(ys) and len(ys) == len(labels)
        for i in range(len(xs)):
            plt.plot(xs[i], ys[i], label=labels[i], linestyle=line_style, marker=marker_style)
        plt.legend()
        if x_label is not None:
            plt.xlabel(x_label)
        if y_label is not None:
            plt.ylabel(y_label)
        
        if x_log:
            plt.xscale('log')
        if y_log:
            plt.yscale('log')

        plt.title(name)
        plt.tight_layout()
        plt.savefig(get_image_path(name), dpi=dpi)
    
    def plot_matsvd(self, mats, descs, name, dpi=100, colormap='Reds', v_min=None, v_max=None):
        log("Plotting {}".format(name))
        if not isinstance(mats[0], list):
            mats = [mats]
            descs = [descs]
        r = mats[0][0].size()[1] / mats[0][0].size()[0]
        plt.figure(figsize=(8 * len(mats), 8 / (r+0.5) * len(mats[0])))
        w_rt = []
        for i in range(len(mats)):
            w_rt += [r, 0.5]
        gs = gridspec.GridSpec(len(mats[0]), 2 * len(mats), width_ratios=w_rt)
        for i in range(len(mats[0])):
            for j in range(len(mats)):
                plt.subplot(gs[i, 2 * j])
                plt.imshow(mats[j][i].to('cpu').numpy(), cmap=colormap, vmin=v_min, vmax=v_max)
                plt.title("{}_{}".format(descs[j][i], name), fontsize=10)
                plt.subplot(gs[i, 2 * j + 1])
                eigenvals = torch.svd(mats[j][i], compute_uv=False)[1].cpu().numpy() # pylint: disable=no-member
                plt.scatter(np.arange(1, len(eigenvals) + 1), eigenvals)
                plt.title('Eigenvalues')
        plt.tight_layout()
        plt.savefig(get_image_path(name), dpi=dpi)

    def plot_matsvd_zero_centered(self, mats, descs, name, dpi=100, colormap='bwr', v_min=None, v_max=None):
        log("Plotting {}".format(name))
        if not isinstance(mats[0], list):
            mats = [mats]
            descs = [descs]
        r = mats[0][0].size()[1] / mats[0][0].size()[0]
        plt.figure(figsize=(8 * len(mats), 8 / (r+0.5) * len(mats[0])))
        w_rt = []
        for i in range(len(mats)):
            w_rt += [r, 0.5]
        gs = gridspec.GridSpec(len(mats[0]), 2 * len(mats), width_ratios=w_rt)
        for i in range(len(mats[0])):
            for j in range(len(mats)):
                plt.subplot(gs[i, 2 * j])
                mat_np = mats[j][i].to('cpu').numpy()
                s_max = np.max(np.abs(mat_np))
                plt.imshow(mat_np, cmap=colormap, vmin=-s_max, vmax=s_max)
                plt.title("{}_{}".format(descs[j][i], name), fontsize=10)

                plt.subplot(gs[i, 2 * j + 1])
                eigenvals = torch.svd(mats[j][i], compute_uv=False)[1].cpu().numpy() # pylint: disable=no-member
                plt.scatter(np.arange(1, len(eigenvals) + 1), eigenvals)
                plt.title('Eigenvalues')
        plt.tight_layout()
        plt.savefig(get_image_path(name), dpi=dpi)
    
    def plot_mat(self, mats, descs, name, dpi=100, w=10, colormap='Reds', v_min=None, v_max=None):
        log("Plotting {}".format(name))
        if not isinstance(mats[0], list):
            mats = [mats]
            descs = [descs]
        r = mats[0][0].size()[1] / mats[0][0].size()[0]
        plt.figure(figsize=(w * len(mats), w / min(r, 3) * len(mats[0])))
        w_rt = []
        for i in range(len(mats)):
            w_rt += [1]
        gs = gridspec.GridSpec(len(mats[0]), len(mats), width_ratios=w_rt)
        for i in range(len(mats[0])):
            for j in range(len(mats)):
                plt.subplot(gs[i, j])
                plt.imshow(mats[j][i].to('cpu').numpy(), cmap=colormap, vmin=v_min, vmax=v_max)
                plt.title("{}_{}".format(descs[j][i], name), fontsize=10)
        plt.tight_layout()
        plt.savefig(get_image_path(name), dpi=dpi)

