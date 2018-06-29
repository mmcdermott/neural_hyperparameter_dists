from abc import ABC, abstractmethod

import random, numpy as np
from scipy.stats import poisson, uniform, beta, geom, dlaplace, rv_discrete

def to_numeric(x): return x[0] if type(x) is np.ndarray else x
def make_rvs(d): return {k: CategoricalRV(v) if type(v) is list else v for k, v in d.items()}
def sample_dict(dict_of_rvs): return {k: to_numeric(v.rvs(1)) for k, v in make_rvs(dict_of_rvs).items()}

class Distribution(ABC):
    def __init__(self):
        """Base Distribution class"""

    def __seed(self, random_state):
        np.random.seed(random_state)
        random.seed(random_state)

    @abstractmethod
    def _sample(self):
        raise NotImplementedError

    def rvs(self, b=1, random_state=None):
        assert b >= 1, "Invalid b: %s" % str(b)
        if random_state is not None: self.__seed(random_state)

        num_samples = b if type(b) is int else b[0]
        return [self._sample() for _ in range(num_samples)]

class MixtureDistribution:
    """ TODO(mmd): Can subclass Distribution?"""
    def __init__(self, candidate_distributions, weights=None):
        self.candidate_distributions = candidate_distributions
        self.num_components = len(self.candidate_distributions)
        self.ws = weights if weights is not None else [1./self.num_components] * self.num_components
        self.distribution_selection = rv_discrete(
            name='mixture_components',
            values=(range(self.num_components), self.ws)
        )

    def rvs(self, b=1, random_state=None):
        assert b >= 1, "Invalid b: %s" % str(b)

        dists = list(self.distribution_selection.rvs(size=b, random_state=random_state))
        counts = [0] * self.num_components
        #print('Num components: ', self.num_components)
        #print('weights: ', self.ws)
        for dist in dists:
            #print(dist)
            counts[dist] += 1

        vals = [None] * self.num_components
        for i, dist, count in zip(range(self.num_components), self.candidate_distributions, counts):
            if count > 0: 
                v = dist.rvs(count, random_state=random_state)
                #print(v)
                vals[i] = [v] if type(v) in [int, np.int32, np.int64, str] else list(v)

        samples = []
        for dist in dists:
            samples += [vals[dist].pop()]
        return samples if b > 1 else samples[0]
        #return samples (for layer dist)

class DictDistribution(Distribution):
    def __init__(self, dict_of_rvs):
        self.dict_of_rvs = dict_of_rvs

    def _sample(self): return sample_dict(self.dict_of_rvs)

def dict_fcr(dict_of_fns):
    return lambda prev, layer_num: DictDistribution(
        {
            k: v(
                prev[k] if type(prev) is dict else prev, layer_num
            ) if type(v) is type(dict_fcr) else v for k, v in dict_of_fns.items()
        }
    )

def build_mult_layer_fcr(base_rv, gamma=1):
    def layer_fn(prev, layer_num):
        if prev is None or gamma == 1.0: return base_rv
        return MixtureDistribution([
            poisson(prev * gamma, loc=2),
            poisson(prev / gamma, loc=2),
            DeltaDistribution(loc=prev),
            poisson(prev, loc=2),
        ])
    return layer_fn

class LayerDistribution(Distribution):
    def __init__(self, num_layer_distribution, layer_distribution_fn):
        self.num_layer_distribution = num_layer_distribution
        self.layer_distribution_fn = layer_distribution_fn

    def _sample(self):
        n = self.num_layer_distribution.rvs(1)[0]
        assert n >= 0, "Invalid number of layers sampled!"

        if n == 0: return []

        layers = [self.layer_distribution_fn(None, 0).rvs(1)[0]]
        for layer in range(1, n): layers.append(self.layer_distribution_fn(layers[-1], layer).rvs(1)[0])

        return layers

class DeltaDistribution(Distribution):
    def __init__(self, loc=0):
        self.x = loc

    def _sample(self): return self.x

class CategoricalRV(MixtureDistribution):
    def __init__(self, options, weights=None):
        super(CategoricalRV, self).__init__([DeltaDistribution(x) for x in options], weights=weights)
