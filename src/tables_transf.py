import os
import sys
from pathlib import Path

import pandas as pd

from exp_configs import Config, csv_sep

if __name__ == '__main__':
    dataset_dir = '../csv.logs/'
    table_dir = '../results/tables/accu/'
    os.makedirs(table_dir,exist_ok=True)
    names = ['dataset_name', 'run', 'step', 'algo', 'active_learn', 'train_size', 'train_time', 'index_time', 'al_time',
             'accuracy', 'micro_prec', 'micro_rec', 'micro_f1', 'macro_prec', 'macro_rec', 'macro_f1']
    configs = ['dataset_name', 'active_learn', 'algo']
    measures = ['accuracy']
    columns = ['step']

    eval_steps = [10, 50, 100, 500, 1000]

    df = pd.DataFrame(columns=names)
    for experiment_set in [['imdb', 'imdb_transf'], ['food', 'food_transf']]:
        for experiment_name in experiment_set:
            for config in [Config.SVM_B_A,Config.PA_O100R_A]:
                for i in range(100):
                    try:
                        df = df.append(
                            pd.read_csv(f'{dataset_dir}/{experiment_name}/{config}_{i}.csv', header=None, names=names,
                                        sep=csv_sep))
                    except:
                        print(experiment_name, config, i)
                        break

    pt = df[df['step'].isin(eval_steps)].pivot_table(index=configs,
                                                     values=measures,
                                                     columns=columns,
                                                     aggfunc='mean')

    pt.to_csv(f'{table_dir}transf_sent_accu.csv', float_format='%.3f')
