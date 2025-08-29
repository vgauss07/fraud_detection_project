from scipy.stats import randint, uniform

LOGISTREG_PARAMS = {
    'penalty': ['l1', 'l2', 'elasticnet'],       # regularization type
    'C': uniform(0.001, 10),                     # inverse regularization strength
    'solver': ['liblinear', 'saga'],             # solvers for classification
    'max_iter': randint(100, 1000),              # iterations for convergence
    'l1_ratio': uniform(0, 1),                   # only used if penalty='elasticnet'
    'class_weight': [None, 'balanced']           # useful for imbalanced classification
}


RANDOM_SEARCH_PARAMS = {
    'n_iter': 5,
    'cv': 5,
    'n_jobs': -1,
    'verbose': 2,
    'random_state': 42,
    'scoring': 'accuracy'
}
