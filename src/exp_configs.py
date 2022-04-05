import sys
from enum import Enum

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

csv_sep = ','


def accuracy_score(test_labels, predictions):
    return np.mean(test_labels == predictions)


def report(test_labels, predictions):
    accuracy = accuracy_score(test_labels, predictions)
    micro_precision = precision_score(test_labels, predictions, average='micro', zero_division=0)
    micro_recall = recall_score(test_labels, predictions, average='micro', zero_division=0)
    micro_f1 = f1_score(test_labels, predictions, average='micro', zero_division=0)

    macro_precision = precision_score(test_labels, predictions, average='macro', zero_division=0)
    macro_recall = recall_score(test_labels, predictions, average='macro', zero_division=0)
    macro_f1 = f1_score(test_labels, predictions, average='macro', zero_division=0)

    return accuracy, micro_precision, micro_recall, micro_f1, macro_precision, macro_recall, macro_f1


class LoggingPrinter:
    def __init__(self, file):
        self.out_file = file
        self.old_stdout = sys.stdout
        sys.stdout = self

    def write(self, text):
        self.old_stdout.write(text)
        self.out_file.write(text)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        sys.stdout = self.old_stdout


class Config(Enum):
    SVM_B_R = 'SVM_B_R'
    PA_O1_R = 'PA_O1_R'
    PA_O10L_R = 'PA_O10L_R'
    PA_O100L_R = 'PA_O100L_R'
    PA_O10R_R = 'PA_O10R_R'
    PA_O100R_R = 'PA_O100R_R'
    SVM_B_A = 'SVM_B_A'
    PA_O1_A = 'PA_O1_A'
    PA_O10L_A = 'PA_O10L_A'
    PA_O100L_A = 'PA_O100L_A'
    PA_O10R_A = 'PA_O10R_A'
    PA_O100R_A = 'PA_O100R_A'
