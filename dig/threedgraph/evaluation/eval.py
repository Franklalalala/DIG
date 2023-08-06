import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch


def plot_rel(label: np.ndarray, prediction: np.ndarray, flag: str, mae: float, r2: float, ):
    fig = plt.figure()

    sns_data = pd.DataFrame({'Prediction': prediction, 'Label': label})

    sns.set_theme(style="darkgrid")
    cmap = sns.light_palette((260, 75, 60), input="husl", as_cmap=True)
    g = sns.jointplot(y='Prediction', x='Label', marker='+', label='Distribution Dots', truncate=False,
                      kind='reg', data=sns_data, palette=cmap,
                      line_kws={'label': 'Regression Line'})

    plt.plot(label, label, label=f"Reference Line", c='y')
    plt.legend(loc='best')
    plt.savefig(f'{flag}_scatter_corr_{r2}_mae_{mae}.png', dpi=400)


def thu_heatmap(label: np.ndarray, prediction: np.ndarray, flag: str, mae: float, r2: float,
                xlim_a=0, xlim_b=0, ylim_a=0, ylim_b=0, x_y_range=1, ):
    from scipy.stats import gaussian_kde

    fig = plt.figure()
    front1 = {'family': 'arial', 'weight': 'normal', 'size': 12}
    front2 = {'family': 'arial', 'weight': 'normal', 'size': 14}
    plt.xlabel(f'Label of {flag}', front1)  # 绘制x轴
    plt.ylabel(f'Prediction of {flag}', front1)  # 绘制y轴
    plt.tick_params(axis='both', which='major', length=6, width=2, direction='in', labelsize=14)  # 设置主坐标轴刻度大小
    plt.tick_params(axis='both', which='minor', length=3, width=1, direction='in', labelsize=14)  # 设置次坐标轴刻度大小

    x = label
    y = prediction
    # Calculate the point density
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    ax = plt.gca()

    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    plt.scatter(x, y, c=z, cmap='Reds', marker='+')
    x_ = [min(label) - x_y_range, max(label) + x_y_range]
    y_ = x_
    plt.plot(x_, y_, ls='--', c='k', alpha=0.5)

    cb1 = plt.colorbar()

    font = {'family': 'Arial',
            'color': 'black',
            'weight': 'normal',
            'size': 10,
            }
    cb1.set_label('Density', fontdict=font)

    if xlim_a != 0 and xlim_b != 0:
        plt.xlim(xlim_a, xlim_b)
    if ylim_a != 0 and ylim_b != 0:
        plt.ylim(ylim_a, ylim_b)
    plt.subplots_adjust(left=0.15)  # 左边距

    # plt.show()
    plt.savefig(f'{flag}_heatmap_corr_{r2}_mae_{mae}.png', dpi=400)


def rename_file(flag):
    os.rename(src='pred', dst=f'{flag}_pred')
    os.rename(src='target', dst=f'{flag}_target')


class ThreeDEvaluator:
    r"""
        Evaluator for the 3D datasets, including QM9, MD17.
        Metric is Mean Absolute Error.
    """

    def __init__(self):
        pass

    def eval(self, input_dict):
        r"""Run evaluation.

        Args:
            input_dict (dict): A python dict with the following items: :obj:`y_true` and :obj:`y_pred`. 
             :obj:`y_true` and :obj:`y_pred` need to be of the same type (either numpy.ndarray or torch.Tensor) and the same shape.

        :rtype: :class:`dict` (a python dict with item :obj:`mae`)
        """
        assert ('y_pred' in input_dict)
        assert ('y_true' in input_dict)

        y_pred, y_true = input_dict['y_pred'], input_dict['y_true']

        assert ((isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray))
                or
                (isinstance(y_true, torch.Tensor) and isinstance(y_pred, torch.Tensor)))
        assert (y_true.shape == y_pred.shape)

        if isinstance(y_true, torch.Tensor):
            return {'mae': torch.mean(torch.abs(y_pred - y_true)).cpu().item()}
        else:
            return {'mae': float(np.mean(np.absolute(y_pred - y_true)))}


class DetailedThreeDEvaluator:
    def __init__(self, dump_info_path: str, info_file_flag: str):
        os.makedirs(dump_info_path, exist_ok=True)
        self.dump_info_path = os.path.abspath(dump_info_path)
        self.info_file_flag = info_file_flag

    def eval(self, input_dict):

        assert ('y_pred' in input_dict)
        assert ('y_true' in input_dict)

        y_pred, y_true = input_dict['y_pred'], input_dict['y_true']

        ########################################################################################################
        from scipy import stats
        cwd_ = os.getcwd()
        os.chdir(self.dump_info_path)
        y_pred_flatten = torch.flatten(y_pred).detach().cpu().numpy()
        y_true_flatten = torch.flatten(y_true).detach().cpu().numpy()
        with open("pred", "w") as pred:
            pred.writelines(list(map(lambda x: str(x) + "\n", y_pred_flatten)))
        with open("target", "w") as true:
            true.writelines(list(map(lambda x: str(x) + "\n", y_true_flatten)))
        r, _ = stats.pearsonr(y_true_flatten, y_pred_flatten)
        mae_e = np.mean(np.abs(y_true_flatten - y_pred_flatten)).round(3)
        plot_rel(label=y_true_flatten, prediction=y_pred_flatten, flag=self.info_file_flag, mae=mae_e, r2=r)
        thu_heatmap(label=y_true_flatten, prediction=y_pred_flatten, flag=self.info_file_flag, mae=mae_e, r2=r)
        rename_file(flag=self.info_file_flag)
        os.chdir(cwd_)
        ########################################################################################################

        assert ((isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray))
                or
                (isinstance(y_true, torch.Tensor) and isinstance(y_pred, torch.Tensor)))
        assert (y_true.shape == y_pred.shape)

        if isinstance(y_true, torch.Tensor):
            return {'mae': torch.mean(torch.abs(y_pred - y_true)).cpu().item()}
        else:
            return {'mae': float(np.mean(np.absolute(y_pred - y_true)))}
