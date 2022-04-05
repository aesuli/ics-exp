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

    for experiment_set in [['imdb', 'imdb_transf'], ['food', 'food_transf']]:
        df = pd.DataFrame(columns=names)
        for experiment_name in experiment_set:
            for config in [Config.SVM_B_A, Config.PA_O100R_A]:
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
            return config[0] == 'accuracy'


        if experiment_set[0].startswith('im'):
            datasetname = 'IMDB'
            sourcename = 'Fine Foods'
        elif experiment_set[0].startswith('fo'):
            datasetname = 'Fine Foods'
            sourcename = 'IMDB'

        if bw:
            monochrome = (
                    cycler('color', ['k'])
                    * cycler('marker', ['', '.', ',', '^'])
                    * cycler('linestyle', ['-', '--', ':', '-.'])
            )
            plt.rcParams['axes.prop_cycle'] = monochrome
        else:
            pal1 = sns.color_palette("Set1", 3)
            sns.set_palette(pal1)

        fig, ax = plt.subplots()
        fig.set_size_inches(5, 5)
        for config in pt.keys():
            if not to_plot(config):
                continue

            y = pt.loc[:, config]
            y = gaussian_filter1d(y, sigma=2)

            if config[2] == 'random':
                linestyle = 'dashed'
            else:
                linestyle = 'solid'

            if config[1].endswith('transf'):
                label = f'Copy({sourcename},{datasetname}) - ' + config[3]
            else:
                label = f'New({datasetname}) - ' + config[3]

            if bw:
                plt.plot(pt.index, y, label=label)
            else:
                plt.plot(pt.index, y, label=label, linestyle=linestyle)

        plt.title(f'Target dataset: {datasetname}')
        if not config[1].startswith('imdb'):
            plt.ylabel('macro F$_1$')
        else:
            plt.ylabel('accuracy')
        plt.xlabel('step')
        plt.legend()

        plot_dir = '../results/plots/accu/'
        os.makedirs(plot_dir, exist_ok=True)
        fig.tight_layout()
        bw_suffix = ''
        if bw:
            bw_suffix = '_bw'
        plt.savefig(plot_dir + 'transf_' + experiment_set[0] + bw_suffix + '.pdf')
