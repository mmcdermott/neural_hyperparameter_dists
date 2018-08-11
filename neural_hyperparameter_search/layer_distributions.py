import scipy.stats as ss, tensorflow as tf

from .distributions import *

# TF Layers Stacks

# TODO(mmd): Finish below
class TFSNNStackDistribution(MarkovianGenerativeProcess):
    def __init__(self): raise NotImplementedError

# TODO(mmd): Need some kind of len generation option.
# TODO(mmd): Need some kind of more general global state control.
class TFDenseStackDistribution(MarkovianGenerativeProcess):
    def __init__(
        self,
        size_rv,
        stop_chance       = 0.5,
        min_depth         = 1,
        max_depth         = 5,
        min_width         = 3,
        max_width         = None,
        activations       = [tf.nn.leaky_relu, tf.nn.relu, tf.nn.elu],
        use_bias          = [True, False],
    ):
        self.min_depth, self.max_depth = min_depth, max_depth
        self.min_width, self.max_width = min_width, max_width
        self.size_rv = to_rv(size_rv)

        self.dist_gen_fn = lambda size_rv: DictDistribution({
            'units':      Censored(size_rv, high_limit=self.max_width, low_limit=self.min_width),
            'activation': activations,
            'use_bias':   use_bias,
        })
        self.should_stop_rv = Coin(stop_chance)

        first_layer = self.dist_gen_fn(self.size_rv)._sample()
        super().__init__(first_layer)

    def _stop(self, prev_state, n):
        if n >= self.max_depth or prev_state['units'] < self.min_width: return True
        if n < self.min_depth: return False

        return self.should_stop_rv._sample()

    def _next(self, prev_state, n):
        return self.dist_gen_fn(self.size_rv)._sample()
