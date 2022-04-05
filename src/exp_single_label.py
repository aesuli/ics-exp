import os
import random
from time import time

import numpy as np
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.svm import LinearSVC

from exp_configs import Config, LoggingPrinter, report, csv_sep


def evaluate_single(step, evals, classifier, test_vects, test_labels, experiment_name, seed, learner_name,
                    al_name, train_vects, train_time, index_time, prediction_time):
    if step in evals:
        if classifier:
            predictions = classifier.predict(test_vects)
        else:
            predictions = test_labels.copy()
            np.random.shuffle(predictions)

        if train_vects is None:
            train_size = 0
        else:
            train_size = train_vects.shape[0]

        print(experiment_name, seed, step, learner_name, al_name, train_size, train_time, index_time, prediction_time,
              *report(test_labels, predictions), sep=csv_sep)


def active_single(classifier, val_sample, val_ids, ids, train_labels):
    if classifier:
        scores = classifier.decision_function(val_sample)
        if len(scores.shape) > 1:
            selection = np.argsort(np.max(scores, axis=1))
        else:
            selection = np.argsort(np.abs(scores))
        for idx in selection:
            if val_ids[idx] not in ids:
                ids.append(val_ids[idx])
                break
    else:
        for idx in range(train_labels.shape[0]):
            if idx not in ids:
                ids.append(idx)
                break


def exp_single_label(csv_dir, seeds, get_dataset_function, vectorizer, experiment_name, config_to_do, evals, steps,
                     average, active_sample_size):
    os.makedirs(f'{csv_dir}/{experiment_name}', exist_ok=True)
    for seed in seeds:
        r = random.Random(seed)
        train_docs, train_labels, test_docs, test_labels = get_dataset_function(r)

        test_vects = vectorizer.fit_transform(test_docs)

        if Config.SVM_B_R in config_to_do:
            with open(f'{csv_dir}/{experiment_name}/{Config.SVM_B_R}_{seed}.csv', mode='tw',
                      encoding='utf-8') as output_file, LoggingPrinter(output_file):
                train_time = 0
                index_time = 0
                prediction_time = 0
                for eval in evals:
                    index_time -= time()
                    try:
                        train_vects = vectorizer.fit_transform(train_docs[:eval])
                    except:
                        train_vects = np.zeros((0, 0))
                    index_time += time()
                    try:
                        train_time -= time()
                        classifier = LinearSVC()
                        classifier.fit(train_vects, train_labels[:eval])
                    except ValueError:
                        classifier = None
                    train_time += time()

                    if classifier:
                        predictions = classifier.predict(test_vects)
                    else:
                        predictions = test_labels.copy()
                        np.random.shuffle(predictions)

                    print(experiment_name, seed, eval, 'SVM', 'random', train_vects.shape[0], train_time, index_time,
                          prediction_time, *report(test_labels, predictions), sep=csv_sep)

        if Config.PA_O1_R in config_to_do:
            with open(f'{csv_dir}/{experiment_name}/{Config.PA_O1_R}_{seed}.csv', mode='tw',
                      encoding='utf-8') as output_file, LoggingPrinter(output_file):
                train_time = 0
                index_time = 0
                prediction_time = 0
                classifier = PassiveAggressiveClassifier(average=average)
                for step in range(steps):
                    index_time -= time()
                    try:
                        train_vects = vectorizer.fit_transform(train_docs[max(step - 1, 0):step])
                    except:
                        train_vects = np.zeros((0, 0))
                    index_time += time()
                    train_time -= time()
                    try:
                        if not classifier:
                            classifier = PassiveAggressiveClassifier(average=average)
                        classifier.partial_fit(train_vects, train_labels[max(step - 1, 0):step],
                                               classes=list(set(train_labels)))
                    except:
                        classifier = None
                    train_time += time()

                    evaluate_single(step, evals, classifier, test_vects, test_labels, experiment_name, seed, f'PA-1',
                                    'random', train_vects, train_time, index_time, prediction_time)

        for config in [Config.PA_O10L_R, Config.PA_O100L_R]:
            if config in config_to_do:
                with open(f'{csv_dir}/{experiment_name}/{config}_{seed}.csv', mode='tw',
                          encoding='utf-8') as output_file, LoggingPrinter(output_file):
                    win = None
                    if config == Config.PA_O10L_R:
                        win = 10
                    elif config == Config.PA_O100L_R:
                        win = 100
                    train_time = 0
                    index_time = 0
                    prediction_time = 0
                    classifier = PassiveAggressiveClassifier(average=average)
                    for step in range(steps):
                        index_time -= time()
                        try:
                            train_vects = vectorizer.fit_transform(train_docs[max(step - win, 0):step])
                        except:
                            train_vects = np.zeros((0, 0))
                        index_time += time()
                        train_time -= time()
                        try:
                            if not classifier:
                                classifier = PassiveAggressiveClassifier(average=average)
                            classifier.partial_fit(train_vects,
                                                   train_labels[max(step - win, 0):step],
                                                   classes=list(set(train_labels)))
                        except:
                            classifier = None
                        train_time += time()

                        evaluate_single(step, evals, classifier, test_vects, test_labels, experiment_name, seed,
                                        f'PA-L-{win}', 'random', train_vects, train_time, index_time, prediction_time)

        for config in [Config.PA_O10R_R, Config.PA_O100R_R]:
            if config in config_to_do:
                with open(f'{csv_dir}/{experiment_name}/{config}_{seed}.csv', mode='tw',
                          encoding='utf-8') as output_file, LoggingPrinter(output_file):
                    win = None
                    if config == Config.PA_O10R_R:
                        win = 10
                    elif config == Config.PA_O100R_R:
                        win = 100
                    train_time = 0
                    index_time = 0
                    prediction_time = 0
                    classifier = PassiveAggressiveClassifier(average=average)
                    for step in range(steps):
                        index_time -= time()
                        try:
                            ids = r.sample(list(range(min(len(train_labels) - 1, step))), min(step, win - 1))
                            ids.append(min(len(train_labels) - 1, step))
                            train_vects = vectorizer.fit_transform(train_docs[ids])
                        except:
                            train_vects = np.zeros((0, 0))
                        index_time += time()
                        train_time -= time()
                        try:
                            if not classifier:
                                classifier = PassiveAggressiveClassifier(average=average)
                            classifier.partial_fit(train_vects, train_labels[ids],
                                                   classes=list(set(train_labels)))
                        except:
                            classifier = None
                        train_time += time()

                        evaluate_single(step, evals, classifier, test_vects, test_labels, experiment_name, seed,
                                        f'PA-R-{win}', 'random', train_vects, train_time, index_time, prediction_time)

        if Config.SVM_B_A in config_to_do:
            with open(f'{csv_dir}/{experiment_name}/{Config.SVM_B_A}_{seed}.csv', mode='tw',
                      encoding='utf-8') as output_file, LoggingPrinter(output_file):
                train_time = 0
                index_time = 0
                prediction_time = 0
                ids = []
                for step in range(steps):
                    index_time -= time()
                    try:
                        train_vects = vectorizer.fit_transform(train_docs[ids])
                    except:
                        train_vects = np.zeros((0, 0))
                    index_time += time()
                    prediction_time -= time()
                    val_ids = np.random.choice(list(range(train_docs.shape[0])), active_sample_size, False)
                    val_sample = vectorizer.fit_transform(train_docs[val_ids])
                    prediction_time += time()
                    try:
                        train_time -= time()
                        classifier = LinearSVC()
                        classifier.fit(train_vects, train_labels[ids])
                    except ValueError:
                        classifier = None
                    train_time += time()
                    prediction_time -= time()
                    active_single(classifier, val_sample, val_ids, ids, train_labels)
                    prediction_time += time()

                    evaluate_single(step, evals, classifier, test_vects, test_labels, experiment_name, seed,
                                    f'SVM', 'active', train_vects, train_time, index_time, prediction_time)

        if Config.PA_O1_A in config_to_do:
            with open(f'{csv_dir}/{experiment_name}/{Config.PA_O1_A}_{seed}.csv', mode='tw',
                      encoding='utf-8') as output_file, LoggingPrinter(output_file):
                train_time = 0
                index_time = 0
                prediction_time = 0
                classifier = PassiveAggressiveClassifier(average=average)
                ids = []
                for step in range(steps):
                    index_time -= time()
                    try:
                        train_vects = vectorizer.fit_transform(train_docs[ids[-1:]])
                    except:
                        train_vects = np.zeros((0, 0))
                    index_time += time()
                    prediction_time -= time()
                    val_ids = np.random.choice(list(range(train_docs.shape[0])), active_sample_size, False)
                    val_sample = vectorizer.fit_transform(train_docs[val_ids])
                    prediction_time += time()
                    train_time -= time()
                    try:
                        if not classifier:
                            classifier = PassiveAggressiveClassifier(average=average)
                        classifier.partial_fit(train_vects, train_labels[ids[-1:]],
                                               classes=list(set(train_labels)))
                    except:
                        classifier = None
                    train_time += time()
                    prediction_time -= time()
                    active_single(classifier, val_sample, val_ids, ids, train_labels)
                    prediction_time += time()

                    evaluate_single(step, evals, classifier, test_vects, test_labels, experiment_name, seed,
                                    f'PA-1', 'active', train_vects, train_time, index_time, prediction_time)

        for config in [Config.PA_O10L_A, Config.PA_O100L_A]:
            if config in config_to_do:
                with open(f'{csv_dir}/{experiment_name}/{config}_{seed}.csv', mode='tw',
                          encoding='utf-8') as output_file, LoggingPrinter(output_file):
                    win = None
                    if config == Config.PA_O10L_A:
                        win = 10
                    elif config == Config.PA_O100L_A:
                        win = 100
                    train_time = 0
                    index_time = 0
                    prediction_time = 0
                    classifier = PassiveAggressiveClassifier(average=average)
                    ids = []
                    for step in range(steps):
                        index_time -= time()
                        try:
                            train_vects = vectorizer.fit_transform(train_docs[ids[-win:]])
                        except:
                            train_vects = np.zeros((0, 0))
                        index_time += time()
                        prediction_time -= time()
                        val_ids = np.random.choice(list(range(train_docs.shape[0])), active_sample_size, False)
                        val_sample = vectorizer.fit_transform(train_docs[val_ids])
                        prediction_time += time()
                        train_time -= time()
                        try:
                            if not classifier:
                                classifier = PassiveAggressiveClassifier(average=average)
                            classifier.partial_fit(train_vects, train_labels[ids[-win:]],
                                                   classes=list(set(train_labels)))
                        except:
                            classifier = None
                        train_time += time()
                        prediction_time -= time()
                        active_single(classifier, val_sample, val_ids, ids, train_labels)
                        prediction_time += time()

                        evaluate_single(step, evals, classifier, test_vects, test_labels, experiment_name, seed,
                                        f'PA-L-{win}', 'active', train_vects, train_time, index_time, prediction_time)

        for config in [Config.PA_O10R_A, Config.PA_O100R_A]:
            if config in config_to_do:
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
                    classifier = PassiveAggressiveClassifier(average=average)
                    ids = []
                    for step in range(steps):
                        index_time -= time()
                        try:
                            sample_ids = r.sample(ids[:-1], min(win - 1, len(ids) - 1))
                            sample_ids.append(ids[-1])
                            train_vects = vectorizer.fit_transform(train_docs[sample_ids])
                        except:
                            train_vects = np.zeros((0, 0))
                        index_time += time()
                        prediction_time -= time()
                        val_ids = np.random.choice(list(range(train_docs.shape[0])), active_sample_size, False)
                        val_sample = vectorizer.fit_transform(train_docs[val_ids])
                        prediction_time += time()
                        train_time -= time()
                        try:
                            if not classifier:
                                classifier = PassiveAggressiveClassifier(average=average)
                            classifier.partial_fit(train_vects, train_labels[sample_ids],
                                                   classes=list(set(train_labels)))
                        except:
                            classifier = None
                        train_time += time()
                        prediction_time -= time()
                        active_single(classifier, val_sample, val_ids, ids, train_labels)
                        prediction_time += time()

                        evaluate_single(step, evals, classifier, test_vects, test_labels, experiment_name, seed,
                                        f'PA-R-{win}', 'active', train_vects, train_time, index_time, prediction_time)
