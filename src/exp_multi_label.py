import os
import random
from time import time

import numpy as np
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.svm import LinearSVC

from exp_configs import Config, LoggingPrinter, report, csv_sep


def evaluate_multi(step, evals, train_labels, classifiers, test_vects, test_labels, experiment_name, learner_name,
                   al_name, seed, train_vects,
                   train_time, index_time, prediction_time):
    if step in evals:
        predictions = list()
        for label in range(train_labels.shape[1]):
            if classifiers[label]:
                label_predictions = classifiers[label].predict(test_vects)
            else:
                label_predictions = test_labels[:, label].copy()
                np.random.shuffle(label_predictions)
            predictions.append(label_predictions)

        predictions = np.vstack(predictions).T

        print(experiment_name, seed, step, learner_name, al_name, train_vects.shape[0], train_time,
              index_time, prediction_time, *report(test_labels, predictions), sep=csv_sep)


def active_multi(classifiers, label, val_sample, val_ids, ids, train_labels):
    if classifiers[label]:
        scores = classifiers[label].decision_function(val_sample)
        if len(scores.shape) > 1:
            selection = np.argsort(np.max(scores, axis=1))
        else:
            selection = np.argsort(np.abs(scores))
        for idx in selection:
            if val_ids[idx] not in ids[label]:
                ids[label].append(val_ids[idx])
                break
    else:
        for idx in range(train_labels.shape[0]):
            if idx not in ids[label]:
                ids[label].append(idx)
                break


def exp_multi_label(csv_dir, seeds, get_dataset_function, vectorizer, experiment_name, config_to_do, evals, steps,
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
                    predictions = list()
                    for label in range(train_labels.shape[1]):
                        try:
                            train_time -= time()
                            classifier = LinearSVC()
                            classifier.fit(train_vects, train_labels[:eval, label])
                        except:
                            classifier = None
                        train_time += time()

                        if classifier:
                            label_predictions = classifier.predict(test_vects)
                        else:
                            label_predictions = test_labels[:, label].copy()
                            np.random.shuffle(label_predictions)
                        predictions.append(label_predictions)

                    predictions = np.vstack(predictions).T

                    print(experiment_name, seed, eval, 'SVM', 'random', train_vects.shape[0], train_time, index_time,
                          prediction_time, *report(test_labels, predictions), sep=csv_sep)

        if Config.PA_O1_R in config_to_do:
            with open(f'{csv_dir}/{experiment_name}/{Config.PA_O1_R}_{seed}.csv', mode='tw',
                      encoding='utf-8') as output_file, LoggingPrinter(output_file):
                train_time = 0
                index_time = 0
                prediction_time = 0
                classifiers = [PassiveAggressiveClassifier(average=average) for _ in range(train_labels.shape[1])]
                for step in range(steps):
                    index_time -= time()
                    try:
                        train_vects = vectorizer.fit_transform(train_docs[max(step - 1, 0):step])
                    except:
                        train_vects = np.zeros((0, 0))
                    index_time += time()
                    for label in range(train_labels.shape[1]):
                        train_time -= time()
                        try:
                            if not classifiers[label]:
                                classifiers[label] = PassiveAggressiveClassifier(average=average)
                            classifiers[label].partial_fit(train_vects, train_labels[max(step - 1, 0):step, label],
                                                           classes=[0, 1])
                        except:
                            classifiers[label] = None
                        train_time += time()

                    evaluate_multi(step, evals, train_labels, classifiers, test_vects, test_labels, experiment_name,
                                   'PA-1', 'random', seed, train_vects, train_time, index_time, prediction_time)

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
                    classifiers = [PassiveAggressiveClassifier(average=average) for _ in range(train_labels.shape[1])]
                    for step in range(steps):
                        index_time -= time()
                        try:
                            train_vects = vectorizer.fit_transform(train_docs[max(step - win, 0):step])
                        except:
                            train_vects = np.zeros((0, 0))
                        index_time += time()
                        for label in range(train_labels.shape[1]):
                            train_time -= time()
                            try:
                                if not classifiers[label]:
                                    classifiers[label] = PassiveAggressiveClassifier(average=average)
                                classifiers[label].partial_fit(train_vects,
                                                               train_labels[max(step - win, 0):step, label],
                                                               classes=[0, 1])
                            except:
                                classifiers[label] = None
                            train_time += time()

                        evaluate_multi(step, evals, train_labels, classifiers, test_vects, test_labels, experiment_name,
                                       f'PA-L-{win}', 'random', seed, train_vects, train_time, index_time,
                                       prediction_time)

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
                    classifiers = [PassiveAggressiveClassifier(average=average) for _ in range(train_labels.shape[1])]
                    for step in range(steps):
                        index_time -= time()
                        try:
                            ids = r.sample(list(range(min(len(train_labels) - 1, step))), min(step, win - 1))
                            ids.append(min(len(train_labels) - 1, step))
                            train_vects = vectorizer.fit_transform(train_docs[ids])
                        except:
                            train_vects = np.zeros((0, 0))
                        index_time += time()
                        for label in range(train_labels.shape[1]):
                            train_time -= time()
                            try:
                                if not classifiers[label]:
                                    classifiers[label] = PassiveAggressiveClassifier(average=average)
                                classifiers[label].partial_fit(train_vects, train_labels[ids, label],
                                                               classes=[0, 1])
                            except:
                                classifiers[label] = None
                            train_time += time()

                        evaluate_multi(step, evals, train_labels, classifiers, test_vects, test_labels, experiment_name,
                                       f'PA-R-{win}', 'random', seed, train_vects, train_time, index_time,
                                       prediction_time)

        if Config.SVM_B_A in config_to_do:
            with open(f'{csv_dir}/{experiment_name}/{Config.SVM_B_A}_{seed}.csv', mode='tw',
                      encoding='utf-8') as output_file, LoggingPrinter(output_file):
                train_time = 0
                index_time = 0
                prediction_time = 0
                ids = list()
                for label in range(train_labels.shape[1]):
                    ids.append([])
                for step in range(steps):
                    prediction_time -= time()
                    val_ids = np.random.choice(list(range(train_docs.shape[0])), active_sample_size, False)
                    val_sample = vectorizer.fit_transform(train_docs[val_ids])
                    prediction_time += time()
                    classifiers = [LinearSVC() for _ in range(train_labels.shape[1])]
                    for label in range(train_labels.shape[1]):
                        index_time -= time()
                        try:
                            train_vects = vectorizer.fit_transform(train_docs[ids[label]])
                        except:
                            train_vects = np.zeros((0, 0))
                        index_time += time()
                        try:
                            train_time -= time()
                            classifiers[label].fit(train_vects, train_labels[ids[label], label])
                        except:
                            classifiers[label] = None
                        train_time += time()
                        prediction_time -= time()
                        active_multi(classifiers, label, val_sample, val_ids, ids, train_labels)
                        prediction_time += time()

                    evaluate_multi(step, evals, train_labels, classifiers, test_vects, test_labels, experiment_name,
                                   f'SVM', 'active', seed, train_vects, train_time, index_time, prediction_time)

        if Config.PA_O1_A in config_to_do:
            with open(f'{csv_dir}/{experiment_name}/{Config.PA_O1_A}_{seed}.csv', mode='tw',
                      encoding='utf-8') as output_file, LoggingPrinter(output_file):
                train_time = 0
                index_time = 0
                prediction_time = 0
                ids = list()
                classifiers = list()
                for label in range(train_labels.shape[1]):
                    ids.append([])
                    classifiers.append(PassiveAggressiveClassifier(average=average))
                for step in range(steps):
                    prediction_time -= time()
                    val_ids = np.random.choice(list(range(train_docs.shape[0])), active_sample_size, False)
                    val_sample = vectorizer.fit_transform(train_docs[val_ids])
                    prediction_time += time()
                    for label in range(train_labels.shape[1]):
                        index_time -= time()
                        try:
                            train_vects = vectorizer.fit_transform(train_docs[ids[label][-1:]])
                        except:
                            train_vects = np.zeros((0, 0))
                        index_time += time()
                        train_time -= time()
                        try:
                            if not classifiers[label]:
                                classifiers[label] = PassiveAggressiveClassifier(average=average)
                            classifiers[label].partial_fit(train_vects, train_labels[ids[label][-1:], label],
                                                           classes=[0, 1])
                        except:
                            classifiers[label] = None
                        train_time += time()
                        prediction_time -= time()
                        active_multi(classifiers, label, val_sample, val_ids, ids, train_labels)
                        prediction_time += time()

                    evaluate_multi(step, evals, train_labels, classifiers, test_vects, test_labels, experiment_name,
                                   f'PA-1', 'active', seed, train_vects, train_time, index_time, prediction_time)

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
                    ids = list()
                    classifiers = list()
                    for label in range(train_labels.shape[1]):
                        ids.append([])
                        classifiers.append(PassiveAggressiveClassifier(average=average))
                    for step in range(steps):
                        prediction_time -= time()
                        val_ids = np.random.choice(list(range(train_docs.shape[0])), active_sample_size, False)
                        val_sample = vectorizer.fit_transform(train_docs[val_ids])
                        prediction_time += time()
                        for label in range(train_labels.shape[1]):
                            index_time -= time()
                            try:
                                train_vects = vectorizer.fit_transform(train_docs[ids[label][-win:]])
                            except:
                                train_vects = np.zeros((0, 0))
                            index_time += time()
                            train_time -= time()
                            try:
                                if not classifiers[label]:
                                    classifiers[label] = PassiveAggressiveClassifier(average=average)
                                classifiers[label].partial_fit(train_vects, train_labels[ids[label][-win:], label],
                                                               classes=[0, 1])
                            except:
                                classifiers[label] = None
                            train_time += time()
                            prediction_time -= time()
                            active_multi(classifiers, label, val_sample, val_ids, ids, train_labels)
                            prediction_time += time()

                        evaluate_multi(step, evals, train_labels, classifiers, test_vects, test_labels, experiment_name,
                                       f'PA-L-{win}', 'active', seed, train_vects, train_time, index_time,
                                       prediction_time)

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
                    ids = list()
                    classifiers = list()
                    for label in range(train_labels.shape[1]):
                        ids.append([])
                        classifiers.append(PassiveAggressiveClassifier(average=average))
                    for step in range(steps):
                        prediction_time -= time()
                        val_ids = np.random.choice(list(range(train_docs.shape[0])), active_sample_size, False)
                        val_sample = vectorizer.fit_transform(train_docs[val_ids])
                        prediction_time += time()
                        for label in range(train_labels.shape[1]):
                            index_time -= time()
                            try:
                                sample_ids = r.sample(ids[label][:-1], min(win - 1, len(ids[label]) - 1))
                                sample_ids.append(ids[label][-1])
                                train_vects = vectorizer.fit_transform(train_docs[sample_ids])
                            except:
                                train_vects = np.zeros((0, 0))
                            index_time += time()
                            train_time -= time()
                            try:
                                if not classifiers[label]:
                                    classifiers[label] = PassiveAggressiveClassifier(average=average)
                                classifiers[label].partial_fit(train_vects, train_labels[sample_ids, label],
                                                               classes=[0, 1])
                            except:
                                classifiers[label] = None
                            train_time += time()
                            prediction_time -= time()
                            active_multi(classifiers, label, val_sample, val_ids, ids, train_labels)
                            prediction_time += time()

                        evaluate_multi(step, evals, train_labels, classifiers, test_vects, test_labels, experiment_name,
                                       f'PA-R-{win}', 'active', seed, train_vects, train_time, index_time,
                                       prediction_time)
