import os

import numpy as np
import pandas as pd
import seaborn as sns
from cycler import cycler
from matplotlib import pyplot as plt

from exp_configs import Config, csv_sep

if __name__ == '__main__':
    dataset_dir = '../csv.logs/'
    plot_dir = '../results/plots/time/'
    os.makedirs(plot_dir, exist_ok=True)
    names = ['dataset_name', 'run', 'step', 'algo', 'active_learn', 'train_size', 'training', 'indexing', 'active_time',
             'accuracy', 'micro_prec', 'micro_rec', 'micro_f1', 'macro_prec', 'macro_rec', 'macro_f1']
    configs = ['dataset_name', 'active_learn', 'algo']
    measures = ['training', 'indexing', 'active_time']

    # bw = False
    bw = True

    for experiment_name in ['imdb', 'tng', 'reut']:
        df = pd.DataFrame(columns=names)
        for config in Config:
            for i in range(100):
                try:
                    expdf = pd.read_csv(f'{dataset_dir}/{experiment_name}/{config}_{i}.csv', header=None, names=names,
                                        sep=csv_sep)
                    df = df.append(expdf)
                except:
                    print(i)
                    break

        pt = df.pivot_table(columns=configs,
                            index='step',
                            values=measures,
                            aggfunc='mean')


        def to_plot(config):
            return config[0] == 'training' and config[2] == 'active' and config[3].find('L') < 0


        if bw:
            monochrome = (
                    cycler('color', ['k'])
                    * cycler('marker', ['', '.', ',', '^'])
                    * cycler('linestyle', ['-', '--', ':', '-.'])
            )
            plt.rcParams['axes.prop_cycle'] = monochrome
        else:
            pal1 = sns.color_palette("Set1", 4)
            sns.set_palette(pal1)

        fig, ax = plt.subplots()
        fig.set_size_inches(4, 3)
        plt.yscale('log')
        for config in pt.keys():
            if not to_plot(config):
                continue
            y = np.asarray(pt.loc[:, config][1:]) - np.asarray(pt.loc[:, config][:-1])
            y += np.asarray(pt.loc[:, tuple(['indexing'] + list(config[1:]))][1:]) - np.asarray(
                pt.loc[:, tuple(['indexing'] + list(config[1:]))][:-1])
            if config[3] == 'SVM' and config[2] == 'random':
                y *= 5
            y /= 5

            if config[2] == 'random':
                linestyle = 'dashed'
            else:
                linestyle = 'solid'

            if bw:
                plt.plot(pt.index[1:], y,
                         label=config[3], markevery=40)
            else:
                plt.plot(pt.index[1:], y,
                         label=config[3], linestyle=linestyle)
        plt.legend()
        plt.ylabel('Model update time')
        if experiment_name.startswith('reut'):
            plt.title(f'Reuters 21578')
        elif experiment_name.startswith('im'):
            plt.title(f'IMDB')
        else:
            plt.title(f'20 Newsgroups')

        plt.xlabel('step')
        # plt.show()

        fig.tight_layout()
        bw_suffix = ''
        if bw:
            bw_suffix = '_bw'
        plt.savefig(plot_dir + experiment_name + '_time' + bw_suffix + '.pdf')
