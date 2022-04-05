import os

import pandas as pd

from exp_configs import Config, csv_sep

if __name__ == '__main__':
    dataset_dir = '../csv.logs/'
    table_dir = '../results/tables/time/'
    os.makedirs(table_dir, exist_ok=True)
    names = ['dataset_name', 'run', 'step', 'algo', 'active_learn', 'train_size', 'training', 'indexing', 'active_time',
             'accuracy', 'micro_prec', 'micro_rec', 'micro_f1', 'macro_prec', 'macro_rec', 'macro_f1']
    configs = ['dataset_name', 'active_learn', 'step', 'algo']
    measures = ['training', 'indexing']

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

        pt = df[df['step'] == 1000].pivot_table(index=configs,
                                                values=measures,
                                                aggfunc='mean')

        final = pt.to_csv(f'{table_dir}{experiment_name}_time.csv', float_format='%.3f')
