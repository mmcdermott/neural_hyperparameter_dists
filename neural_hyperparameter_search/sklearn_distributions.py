import scipy.stats as ss

from .distributions import *

LR_dist = DictDistribution({
    'penalty': ['l1', 'l2'],
    'class_weight': [None, 'balanced'],
    'C': ss.uniform(1e-4, 10),
})

RF_dist = DictDistribution({
    'n_estimators': [ss.poisson(10), ss.poisson(50), ss.poisson(100), ss.poisson(250), ss.poisson(500)],
    'criterion': ['gini', 'entropy'],
    'max_depth': ss.randint(2, 10),
    'min_samples_split': ss.randint(2, 75),
    'min_samples_leaf': ss.randint(1, 50),
})

MLP_dist = DictDistribution({
    'hidden_layer_sizes': ListRV(
        el_rv=[ss.poisson(25), ss.poisson(50), ss.poisson(250)],
    ),
    'activation': ['logistic', 'tanh', 'relu'],
    'solver': ['lbfgs', 'sgd', 'adam'],
    'alpha': ss.uniform(1e-6, 1.5),
    'batch_size': ['auto', ss.randint(16, 512)],
    'learning_rate': ['constant', 'invscaling', 'adaptive'],
    'learning_rate_init': ss.uniform(1e-5, 1),
    'power_t': ss.beta(5, 5),
    'max_iter': ss.randint(50, 700),
    'momentum': ss.beta(9, 1),
    'early_stopping': [True, False], # Coin(p) gives weighted
    'validation_fraction': ss.beta(1, 9),
})
