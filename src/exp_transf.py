import os
import random
from time import time

import numpy as np
from sklearn.linear_model import PassiveAggressiveClassifier

from exp_configs import Config, LoggingPrinter
from exp_single_label import active_single, evaluate_single


def exp_transf(csv_dir, experiment_name, learner_name, al_name, seeds, get_dataset_source_function,
               get_dataset_target_function, vectorizer, config, average, steps, active_sample_size, evals):
    os.makedirs(f'{csv_dir}/{experiment_name}', exist_ok=True)
    for seed in seeds:
        r = random.Random(seed)

        source_train_docs, source_train_labels, source_test_docs, source_test_labels = get_dataset_source_function(r)

        win = None
        if config == Config.PA_O10R_A:
            win = 10
        elif config == Config.PA_O100R_A:
            win = 100
        classifier = PassiveAggressiveClassifier(average=average)
        ids = []
        for step in range(steps):
            try:
                sample_ids = r.sample(ids[:-1], min(win - 1, len(ids) - 1))
                sample_ids.append(ids[-1])
                train_vects = vectorizer.fit_transform(source_train_docs[sample_ids])
            except:
                train_vects = np.zeros((0, 0))
            val_ids = np.random.choice(list(range(source_train_docs.shape[0])), active_sample_size, False)
            val_sample = vectorizer.fit_transform(source_train_docs[val_ids])
            try:
                if not classifier:
                    classifier = PassiveAggressiveClassifier(average=average)
                classifier.partial_fit(train_vects, source_train_labels[sample_ids],
                                       classes=list(set(source_train_labels)))
            except:
                classifier = None
            active_single(classifier, val_sample, val_ids, ids, source_train_labels)

        train_docs, train_labels, test_docs, test_labels = get_dataset_target_function(r)

        test_vects = vectorizer.fit_transform(test_docs)
        with open(f'{csv_dir}/{experiment_name}/{config}_{seed}.csv', mode='tw',
                  encoding='utf-8') as output_file, LoggingPrinter(output_file):
            win = None
            if config == Config.PA_O10R_A:
                win = 10
            elif config == Config.PA_O100R_A:
                win = 100
            train_time = 0
            index_time = 0
            prediction_time = 0
            ids = []
            for step in range(steps):
                index_time -= time()
                try:
                    sample_ids = r.sample(ids[:-1], min(win - 1, len(ids) - 1))
                    sample_ids.append(ids[-1])
                    train_vects = vectorizer.fit_transform(train_docs[sample_ids])
                except:
                    train_vects = None
                index_time += time()
                prediction_time -= time()
                val_ids = np.random.choice(list(range(train_docs.shape[0])), active_sample_size, False)
                val_sample = vectorizer.fit_transform(train_docs[val_ids])
                prediction_time += time()
                train_time -= time()
                if train_vects is not None:
                    classifier.partial_fit(train_vects, train_labels[sample_ids],
                                           classes=list(set(train_labels)))
                train_time += time()
                prediction_time -= time()
                active_single(classifier, val_sample, val_ids, ids, train_labels)
                prediction_time += time()

                evaluate_single(step, evals, classifier, test_vects, test_labels, experiment_name, seed,
                                learner_name, al_name, train_vects, train_time, index_time, prediction_time)
