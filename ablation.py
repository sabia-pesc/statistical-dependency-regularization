import pandas as pd
from datetime import datetime
from aif360.metrics import ClassificationMetric
from aif360.algorithms.inprocessing import PrejudiceRemover, AdversarialDebiasing, MetaFairClassifier, GerryFairClassifier
from models import (FairTransitionLossMLP, SimpleMLP, describe_metrics, HIFIClassifier)
from fitness_rules import *
from dataset_readers import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow.compat.v1 as tf_old
#tf_old.disable_eager_execution()

import numpy as np
import gc
import os
import multiprocessing
from pathlib import Path
import optuna
from util import mathew_correlation_coefficient, f1_score
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler

N_TRIALS = 100
DEFAULT_N_JOBS = 8
SAMPLER = TPESampler
PRUNER = HyperbandPruner
PROJECT_ROOT = Path(__file__).parent.absolute()
CONNECTION_STRING = os.environ.get('CONNECTION_STRING')
if CONNECTION_STRING is None:
    CONNECTION_STRING = f'sqlite:///{PROJECT_ROOT}/optuna_study.db'
STORAGE = optuna.storages.RDBStorage(
    url=CONNECTION_STRING,
    engine_kwargs={'connect_args': {'timeout': 30}} if CONNECTION_STRING.startswith('sqlite') else {}
)
start_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')

def get_n_jobs():
    env_n_jobs = os.getenv('OPTUNA_N_JOBS')
    if env_n_jobs is not None:
        return int(env_n_jobs)
    else:
        return DEFAULT_N_JOBS

def get_sampler():
    return TPESampler()

def get_pruner():
    return HyperbandPruner()

def eval(model, dataset, unprivileged_groups, privileged_groups, fitness_rule, hyperparameters):
    try:
        # sklearn classifier
        y_pred_prob = model.predict_proba(dataset.features)
        pos_ind = np.where(model.classes_ == dataset.favorable_label)[0][0]
        y_pred = (y_pred_prob[:, 1] > 0.5).astype(np.float64)


        y_pred_mapped = y_pred.copy()
        # Map the dataset labels to back to their original values.
        y_pred_mapped[y_pred == 0] = dataset.unfavorable_label
        y_pred_mapped[y_pred == 1] = dataset.favorable_label

        dataset_pred = dataset.copy()
        dataset_pred.labels = y_pred_mapped

    except AttributeError:
        # aif360 inprocessing algorithm
        y_pred = model.predict(dataset).labels

        dataset_pred = dataset.copy()
        dataset_pred.labels = y_pred

        # Map the dataset labels to back to their original values.
        temp_labels = dataset_pred.labels.copy()

        temp_labels[(dataset_pred.labels == 1.0).ravel(), 0] = dataset.favorable_label
        temp_labels[(dataset_pred.labels == 0.0).ravel(), 0] = dataset.unfavorable_label

        dataset_pred.labels = temp_labels.copy()
    metric = ClassificationMetric(
            dataset, dataset_pred,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups)

    metrics = dict()
    metrics['overall_acc'] = abs(metric.accuracy())
    metrics['bal_acc'] = abs((metric.true_positive_rate()
                                 + metric.true_negative_rate()) / 2)
    metrics['avg_odds_diff'] = metric.average_abs_odds_difference()
    metrics['disp_imp'] = abs(metric.disparate_impact())
    metrics['stat_par_diff'] = abs(metric.statistical_parity_difference())
    metrics['eq_opp_diff'] = abs(metric.equal_opportunity_difference())
    metrics['theil_ind'] = abs(metric.theil_index())
    metrics.update(metric.performance_measures())
    metrics['MCC'] = mathew_correlation_coefficient(metrics)
    metrics['f1_score'] = f1_score(metrics)
    metrics['fitness'] = fitness_rule(metrics)
    if type(hyperparameters) is not dict:
        metrics['solution'] = hyperparameters.params
    else:
        metrics['solution'] = hyperparameters
    metrics['corr'] = model.corr
    return metrics


class WeightedFitnessRule:
    """Picklable fitness rule (replaces a closure so it can cross process boundaries)."""

    def __init__(self, performance_metric, fairness_metric, alpha):
        self.performance_metric = performance_metric
        self.fairness_metric = fairness_metric
        self.alpha = alpha

    def __call__(self, metrics):
        performance = metrics[self.performance_metric]
        fairness = metrics[self.fairness_metric]
        return self.alpha * performance - (1 - self.alpha) * fairness


def _optimize_worker(study_name, n_trials_worker, dataset_reader, model_initializer, fitness_rule):
    """Runs in its own OS process: builds its own dataset copy and TF/Keras graph,
    then reports n_trials_worker trials into the shared (SQLite-backed) Optuna study.
    Exiting the process fully releases its TF/Keras memory before the next worker batch."""
    dataset_expanded_train, dataset_train, dataset_val, dataset_test, unprivileged_groups, privileged_groups, sens_attr = dataset_reader()

    scaler = StandardScaler()
    dataset_train.features = scaler.fit_transform(dataset_train.features)
    dataset_val.features = scaler.transform(dataset_val.features)

    def objective(trial):
        trial_model = model_initializer(sens_attr, unprivileged_groups, privileged_groups, hyperparameters=trial, fitness_rule=fitness_rule)
        trial_model = trial_model.fit(dataset_train.copy(), verbose=False)
        result = eval(trial_model, dataset_val.copy(), unprivileged_groups, privileged_groups, fitness_rule, trial)
        return result['fitness']

    study = optuna.load_study(study_name=study_name, storage=STORAGE,
                               sampler=get_sampler(), pruner=get_pruner())
    study.optimize(objective, n_trials=n_trials_worker)


def tune_model(dataset_reader, model_initializer, fitness_rule, fitness_rule_name, alpha=None, fixed_lambda=None):
    dataset_expanded_train, dataset_train, dataset_val, dataset_test, unprivileged_groups, privileged_groups, sens_attr = dataset_reader()

    scaler = StandardScaler()
    dataset_expanded_train.features = scaler.fit_transform(dataset_expanded_train.features)
    dataset_test.features = scaler.transform(dataset_test.features)

    if fitness_rule is not None:
        # best solution
        now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        study_name = "{0}_{1}_{2}".format(fitness_rule_name, model_initializer.__name__, now)

        study = optuna.create_study(direction='maximize',
                                    study_name=study_name,
                                    pruner=get_pruner(),
                                    sampler=get_sampler(),
                                    storage=STORAGE)

        N_JOBS = get_n_jobs()
        print(f"optimizing with {N_JOBS} worker processes")

        trials_per_worker = [N_TRIALS // N_JOBS] * N_JOBS
        for i in range(N_TRIALS % N_JOBS):
            trials_per_worker[i] += 1

        processes = []
        for n_trials_worker in trials_per_worker:
            if n_trials_worker == 0:
                continue
            p = multiprocessing.Process(
                target=_optimize_worker,
                args=(study_name, n_trials_worker, dataset_reader, model_initializer, fitness_rule)
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        study = optuna.load_study(study_name=study_name, storage=STORAGE)

        # eval on test set
        model = model_initializer(sens_attr, unprivileged_groups, privileged_groups, hyperparameters=study.best_params, fitness_rule=fitness_rule, fixed_lambda=fixed_lambda)
    else:
        model = model_initializer(sens_attr, unprivileged_groups, privileged_groups, fitness_rule=fitness_rule, fixed_lambda=fixed_lambda)

    model = model.fit(dataset_expanded_train, verbose=False)
    best_result = eval(model, dataset_test, unprivileged_groups, privileged_groups, fitness_rule, study.best_params)

    best_result['tune_results_history'] = study.trials_dataframe().to_dict('records')
    if fitness_rule is not None:
        best_result['fitness_rule'] = fitness_rule_name
    else:
        best_result['fitness_rule'] = 'No optimization'
    if alpha is not None:
        best_result['alpha'] = alpha
    if fixed_lambda is not None:
        best_result['fixed_lambda'] = fixed_lambda

    print('-----------------------------------')
    describe_metrics(best_result)
    best_result['method'] = model_initializer.__name__
    best_result['dataset'] = dataset_reader.__name__
    print('-----------------------------------')

    try:
        # tk classifier
        best_result['best_solution_tf_history'] = model.history.history
    except AttributeError:
        # aif360 inprocessing algorithm
        best_result['best_solution_tf_history'] = None

    return best_result


def ftl_mlp_xi_reg_initializer(sens_attr, unprivileged_groups, privileged_groups, hyperparameters=None, fitness_rule=None, fixed_lambda=None):
    hidden_sizes = [100, 100]
    corr_type = 'xi'
    if type(hyperparameters) is not dict:
        if fixed_lambda is not None:
            l2 = fixed_lambda
        else:
            l2 = hyperparameters.suggest_categorical('l2', [1e-2, 1e-3, 1e-4])
        dropout = hyperparameters.suggest_float('dropout', 0.0, 0.2)
        privileged_demotion = hyperparameters.suggest_float('privileged_demotion', 0.0, 1.0)
        privileged_promotion = hyperparameters.suggest_float('privileged_promotion', 0.0, 1.0)
        protected_demotion = hyperparameters.suggest_float('protected_demotion', 0.0, 1.0)
        protected_promotion = hyperparameters.suggest_float('protected_promotion', 0.0, 1.0)
    else:
        l2 = hyperparameters['l2']
        dropout = hyperparameters['dropout']
        privileged_demotion = hyperparameters['privileged_demotion']
        privileged_promotion = hyperparameters['privileged_promotion']
        protected_demotion = hyperparameters['protected_demotion']
        protected_promotion = hyperparameters['protected_promotion']

    if hyperparameters is not None:
        model = FairTransitionLossMLP(sensitive_attr=sens_attr,
                                      hidden_sizes=hidden_sizes,
                                      dropout=dropout,
                                      batch_size=64,
                                      privileged_demotion=privileged_demotion,
                                      privileged_promotion=privileged_promotion,
                                      protected_demotion=protected_demotion,
                                      protected_promotion=protected_promotion,
                                      corr_type=corr_type, l2=l2)
    else:
        model = FairTransitionLossMLP(sensitive_attr=sens_attr,
                                      hidden_sizes=[32],
                                      dropout=0.1,
                                      batch_size=64)
    return model

def simple_mlp_initializer(sens_attr, unprivileged_groups, privileged_groups, hyperparameters=None, fitness_rule=None, fixed_lambda=None):
    hidden_sizes = [100, 100]
    corr_type = None
    l2 = 0.0
    if type(hyperparameters) is not dict:
        dropout = hyperparameters.suggest_float('dropout', 0.0, 0.2)
    else:
        dropout = hyperparameters['dropout']
    if hyperparameters is not None:

        model = SimpleMLP(sensitive_attr=sens_attr,
                        hidden_sizes=hidden_sizes,
                        dropout=dropout,
                        batch_size=64,
                        corr_type=corr_type,
                        l2=l2)
    else:
        model = SimpleMLP(sensitive_attr=sens_attr,
                        hidden_sizes=[32],
                        dropout=0.1,
                        batch_size=64)
    return model

def ftl_mlp_initializer(sens_attr, unprivileged_groups, privileged_groups, hyperparameters=None, fitness_rule=None, fixed_lambda=None):
    hidden_sizes = [100,100]
    corr_type = None
    l2 = 0.0
    if type(hyperparameters) is not dict:
        dropout = hyperparameters.suggest_float('dropout', 0.0, 0.2)
        privileged_demotion = hyperparameters.suggest_float('privileged_demotion', 0.0, 1.0)
        privileged_promotion = hyperparameters.suggest_float('privileged_promotion', 0.0, 1.0)
        protected_demotion = hyperparameters.suggest_float('protected_demotion', 0.0, 1.0)
        protected_promotion = hyperparameters.suggest_float('protected_promotion', 0.0, 1.0)
    else:
        dropout = hyperparameters['dropout']
        privileged_demotion = hyperparameters['privileged_demotion']
        privileged_promotion = hyperparameters['privileged_promotion']
        protected_demotion = hyperparameters['protected_demotion']
        protected_promotion = hyperparameters['protected_promotion']

    if hyperparameters is not None:
        model = FairTransitionLossMLP(sensitive_attr=sens_attr,
                                      hidden_sizes=hidden_sizes,
                                      dropout=dropout,
                                      batch_size=64,
                                      privileged_demotion=privileged_demotion,
                                      privileged_promotion=privileged_promotion,
                                      protected_demotion=protected_demotion,
                                      protected_promotion=protected_promotion,
                                      corr_type=corr_type, l2=l2)
    else:
        model = FairTransitionLossMLP(sensitive_attr=sens_attr,
                                      hidden_sizes=[32],
                                      dropout=0.1,
                                      batch_size=64)
    return model

def mlp_xi_reg_initializer(sens_attr, unprivileged_groups, privileged_groups, hyperparameters=None, fitness_rule=None, fixed_lambda=None):
    hidden_sizes = [100, 100]
    corr_type = 'xi'
    if type(hyperparameters) is not dict:
        if fixed_lambda is not None:
            l2 = fixed_lambda
        else:
            l2 = hyperparameters.suggest_categorical('l2', [1e-2, 1e-3, 1e-4])
        dropout = hyperparameters.suggest_float('dropout', 0.0, 0.2)
    else:
        l2 = hyperparameters['l2']
        dropout = hyperparameters['dropout']
    if hyperparameters is not None:

        model = SimpleMLP(sensitive_attr=sens_attr,
                          hidden_sizes=hidden_sizes,
                          dropout=dropout,
                          batch_size=64,
                          corr_type=corr_type,
                          l2=l2)
    else:
        model = SimpleMLP(sensitive_attr=sens_attr,
                          hidden_sizes=[32],
                          dropout=0.1,
                          batch_size=64)
    return model


def hifi_initializer(sens_attr, unprivileged_groups, privileged_groups, hyperparameters=None, fitness_rule=None, fixed_lambda=None):
    classifier_name = 'dl'
    if type(hyperparameters) is not dict:
        eta = hyperparameters.suggest_float('eta', 1e-3, 1000.0, log=True)
    else:
        eta = hyperparameters['eta']
    if hyperparameters is not None:
        model = HIFIClassifier(sensitive_attr=sens_attr,
                               classifier_name=classifier_name,
                               eta=eta,
                               batch_size=64)
    else:
        model = HIFIClassifier(sensitive_attr=sens_attr,
                               classifier_name=classifier_name,
                               batch_size=64)
    return model

datasets = [
    #adult_dataset_reader,
    bank_dataset_reader,
    #compas_dataset_reader,
    #german_dataset_reader
]

methods = [
    ftl_mlp_xi_reg_initializer,
    #simple_mlp_initializer,
    #mlp_xi_reg_initializer,
    #ftl_mlp_initializer
    hifi_initializer
]

def alpha_ablation():
    results = []

    for dataset_reader in datasets:
        for model_initializer in methods:
            for performance_metric in ['overall_acc', 'MCC']:
                for fairness_metric in ['stat_par_diff', 'avg_odds_diff', 'eq_opp_diff']:
                    for alpha in [0.2, 0.35, 0.5, 0.65, 0.8]:

                        alpha_path_name = str(alpha).replace('.','')
                        fitness_rule_name = f'alpha_{alpha_path_name}_{performance_metric}_{fairness_metric}'
                        fitness_rule = WeightedFitnessRule(performance_metric, fairness_metric, alpha)

                        result = tune_model(dataset_reader, model_initializer, fitness_rule, fitness_rule_name, alpha=alpha)
                        print('Best metrics')
                        print('Dataset:', dataset_reader.__name__)
                        print('Method:', model_initializer.__name__)
                        print('Fitness rule:', fitness_rule_name)

                        describe_metrics(result)
                        results.append(result)
                        results_df = pd.DataFrame(results)
                        results_df.to_csv('raw_results/results_%s.csv' % start_time)
                        gc.collect()
    

def lambda_ablation():
    results = []

    for dataset_reader in datasets:
        for model_initializer in methods:
            for performance_metric in ['overall_acc', 'MCC']:
                for fairness_metric in ['stat_par_diff', 'avg_odds_diff', 'eq_opp_diff']:
                    for fixed_lambda in [1e-2, 1e-3, 1e-4]:

                        lambda_path_name = str(fixed_lambda)
                        fitness_rule_name = f'fixed_lambda_{lambda_path_name}_{performance_metric}_{fairness_metric}'
                        fitness_rule = WeightedFitnessRule(performance_metric, fairness_metric, 0.5)

                        result = tune_model(dataset_reader, model_initializer, fitness_rule, fitness_rule_name, fixed_lambda=fixed_lambda)
                        print('Best metrics')
                        print('Dataset:', dataset_reader.__name__)
                        print('Method:', model_initializer.__name__)
                        print('Fitness rule:', fitness_rule_name)

                        describe_metrics(result)
                        results.append(result)
                        results_df = pd.DataFrame(results)
                        results_df.to_csv('raw_results/results_%s.csv' % start_time)
                        gc.collect()



    

def main():
    alpha_ablation()
    #lambda_ablation()


if __name__ == '__main__':
    main()
