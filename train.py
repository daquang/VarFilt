#!/usr/bin/env python3
"""
Script for training model.
Use `train.py -h` to see an auto-generated description of advanced options.
"""

import argparse
import pickle as pkl

import numpy as np
from scipy.sparse import vstack
from sklearn.datasets import load_svmlight_files
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def get_args():
    parser = argparse.ArgumentParser(description="Train model.",
                                     epilog='\n'.join(__doc__.strip().split('\n')[1:]).strip(),
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-i', '--input', required=True,
                        help='Training SVMlight format file(s)', type=str, nargs='+',)
    parser.add_argument('-j', '--valid', required=False, default=None,
                        help='Testing SVMlight format file(s)', type=str, nargs='*',)
    parser.add_argument('-k', '--test', required=False, default=None,
                        help='Validation SVMlight format file(s)', type=str, nargs='*',)
    parser.add_argument('-p', '--processes',
                        help='Number of parallel processes (default: -1, i.e. all cores).',
                        type=int, default=-1)
    parser.add_argument('-k', '--kfolds',
                        help='Number of k-folds for cross validation (default: 5).',
                        type=int, default=5)
    parser.add_argument('-m', '--model',
                        help='Model (default: rf)',
                        type=str, choices=['rf', 'svm-rbf', 'lr'], default='rf')
    parser.add_argument('-s', '--seed',
                        help='Random seed for reproducibility (default: 1337).',
                        type=int, default=1337)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-o', '--outputdir', type=str,
                       help='The output directory. Causes error if the directory already exists.')
    group.add_argument('-oc', '--outputdirc', type=str,
                       help='The output directory. Will overwrite if directory already exists.')
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    # Load training data
    data_train = load_svmlight_files(args.input)
    X_train = vstack(data_train[0::2]).toarray()
    y_train = vstack(data_train[1::2]).toarray()

    # Make model
    if args.model == 'rf':
        model = RandomForestClassifier()
        param_grid = rf_param_grid()
    elif args.model == 'svm_rbf':
        model = SVC()
        param_grid = svm_rbf_param_grid()

    # Grid search hyperparameters
    grid_search = GridSearchCV(estimator=model, scoring='average_precision', param_grid=param_grid,
                               cv=KFold(len(X_train), n_folds=args.kfolds, shuffle=True, random_state=args.seed),
                               n_jobs=args.processes, verbose=2)

    grid_search.fit(X_train, y_train)

    pkl.dump(grid_search, open('temp.pkl', 'wb'))


def rf_param_grid():
    param_grid = {
        'max_depth': [None, 1, 2, 5, 10, 20, 30, 40, 50],
        'max_features': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        'min_samples_leaf': [0.1, 0.2, 0.3, 0.4, 0.5],
        'min_samples_split': [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8],
        'n_estimators': [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    }
    return param_grid


def svm_rbf_param_grid():
    C_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-9, 3, 13)
    param_grid = dict(gamma=gamma_range, C=C_range)
    return param_grid


if __name__ == '__main__':
    main()