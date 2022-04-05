from datasets import get_reut
from exp_configs import Config
from exp_multi_label import exp_multi_label
from ics.classifier.lri import LightweightRandomIndexingVectorizer

if __name__ == '__main__':
    seeds = list(range(10))
    steps = 1001
    eval_mod = 5
    evals = [i for i in range(steps) if i % eval_mod == 0]

    n_features = 2 ** 13
    average = False
    vectorizer = LightweightRandomIndexingVectorizer(n_features=n_features)
    active_sample_size = 1000

    config_to_do = [
        Config.SVM_B_R,
        Config.PA_O1_R,
        Config.PA_O10L_R,
        Config.PA_O100L_R,
        Config.PA_O10R_R,
        Config.PA_O100R_R,
        Config.SVM_B_A,
        Config.PA_O1_A,
        Config.PA_O10L_A,
        Config.PA_O100L_A,
        Config.PA_O10R_A,
        Config.PA_O100R_A,
    ]

    csv_dir = f'../csv.logs/'

    get_dataset_function = get_reut

    experiment_name = 'reut'
    average = False
    exp_multi_label(csv_dir, seeds, get_dataset_function, vectorizer, experiment_name, config_to_do, evals, steps,
                    average, active_sample_size)
