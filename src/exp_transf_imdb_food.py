from datasets import get_imdb, get_fine_food
from exp_configs import Config
from exp_single_label import exp_single_label
from exp_transf import exp_transf
from ics.classifier.lri import LightweightRandomIndexingVectorizer

if __name__ == '__main__':

    seeds = list(range(10))
    steps = 1001
    eval_mod = 5
    evals = [i for i in range(steps) if i % eval_mod == 0]

    n_features = 2 ** 13
    vectorizer = LightweightRandomIndexingVectorizer(n_features=n_features)
    active_sample_size = 1000

    config = Config.PA_O100R_A

    csv_dir = f'../csv.logs/'

    average = False

    if config == Config.PA_O10R_A:
        learner_name = 'PA-R-10'
    elif config == Config.PA_O100R_A:
        learner_name = 'PA-R-100'

    al_name = 'active'

    get_dataset_target_function = get_imdb
    get_dataset_source_function = get_fine_food
    experiment_name = 'imdb_transf'

    exp_transf(csv_dir, experiment_name, learner_name, al_name, seeds, get_dataset_source_function,
               get_dataset_target_function, vectorizer, config, average, steps, active_sample_size, evals)

    get_dataset_target_function = get_fine_food
    get_dataset_source_function = get_imdb
    experiment_name = 'food_transf'

    exp_transf(csv_dir, experiment_name, learner_name, al_name, seeds, get_dataset_source_function,
               get_dataset_target_function, vectorizer, config, average, steps, active_sample_size, evals)

    config_to_do = [
        Config.SVM_B_A,
        Config.PA_O100R_A,
    ]

    experiment_name = 'food'
    average = False
    exp_single_label(csv_dir, seeds, get_fine_food, vectorizer, experiment_name, config_to_do, evals, steps,
                     average, active_sample_size)