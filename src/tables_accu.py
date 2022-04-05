import os

import pandas as pd

from exp_configs import Config, csv_sep

if __name__ == '__main__':
    dataset_dir = '../csv.logs/'
    table_dir = '../results/tables/accu/'
    os.makedirs(table_dir,exist_ok=True)
    names = ['dataset_name', 'run', 'step', 'algo', 'active_learn', 'train_size', 'train_time', 'index_time', 'al_time',
             'accuracy', 'micro_prec', 'micro_rec', 'micro_f1', 'macro_prec', 'macro_rec', 'macro_f1']
    configs = ['dataset_name', 'active_learn', 'step', 'algo']
    eval_steps = [1000]

    for experiment_name in ['imdb', 'tng', 'reut']:
        if experiment_name in ['imdb', 'tng']:
            measures = ['accuracy', 'macro_prec', 'macro_rec', 'macro_f1']
        else:
            measures = ['micro_prec', 'micro_rec', 'micro_f1', 'macro_prec', 'macro_rec', 'macro_f1']
        df = pd.DataFrame(columns=names)
        for config in Config:
            for i in range(100):
                try:
                    df = df.append(
                        pd.read_csv(f'{dataset_dir}/{experiment_name}/{config}_{i}.csv', header=None, names=names, sep=csv_sep))
                except:
                    print(experiment_name, config, i)
                    break

        pt = df[df['step'].isin(eval_steps)].pivot_table(index=configs,
                                                         values=measures,
                                                         aggfunc='mean')

        pt.to_csv(f'{table_dir}{experiment_name}_accu.csv', float_format='%.3f')
