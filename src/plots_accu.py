import os

import pandas as pd
import seaborn as sns
from cycler import cycler
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d

from exp_configs import Config, csv_sep

if __name__ == '__main__':
    dataset_dir = '../csv.logs/'

    names = ['dataset_name', 'run', 'step', 'algo', 'active_learn', 'train_size', 'training', 'indexing', 'active',
             'accuracy', 'micro_prec', 'micro_rec', 'micro_f1', 'macro_prec', 'macro_rec', 'macro_f1']
    configs = ['dataset_name', 'active_learn', 'algo']
    measures = ['accuracy', 'micro_prec', 'micro_rec', 'micro_f1', 'macro_prec', 'macro_rec', 'macro_f1']

    # bw = False
    bw = True

    for experiment_name in ['imdb', 'tng', 'reut']:
        df = pd.DataFrame(columns=names)
        for config in Config:
            for i in range(100):
                try:
                    df = df.append(
                        pd.read_csv(f'{dataset_dir}/{experiment_name}/{config}_{i}.csv', header=None, names=names,
                                    sep=csv_sep))
                except:
                    print(experiment_name, config, i)
                    break

        pt = df.pivot_table(columns=configs,
                            index='step',
                            values=measures,
                            aggfunc='mean')


        def to_plot(config):
            return ((not config[1].startswith('imdb') and config[0] == 'macro_f1') or (
                        config[1].startswith('imdb') and config[0] == 'accuracy')) and config[2] == 'active' and config[
                       3].find('L') < 0


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
        for config in pt.keys():
            if not to_plot(config):
                continue

            y = pt.loc[:, config]
            y = gaussian_filter1d(y, sigma=2)

            if config[2] == 'random':
                linestyle = 'dashed'
            else:
                linestyle = 'solid'

            if bw:
                plt.plot(pt.index, y,
                         label=config[3], markevery=40)
            else:
                plt.plot(pt.index, y,
                         label=config[3], linestyle=linestyle)

        if experiment_name.startswith('reut'):
            plt.ylabel('macro-F$_1$')
            plt.title(f'Reuters 21578')
        else:
            plt.ylabel('accuracy')
            if experiment_name.startswith('im'):
                plt.title(f'IMDB')
            else:
                plt.title(f'20 Newsgroups')
        plt.legend()

        plt.xlabel('step')
        # plt.show()
        plot_dir = '../results/plots/accu/'
        os.makedirs(plot_dir, exist_ok=True)
        fig.tight_layout()
        bw_suffix = ''
        if bw:
            bw_suffix = '_bw'
        plt.savefig(plot_dir + experiment_name + bw_suffix + '.pdf')
