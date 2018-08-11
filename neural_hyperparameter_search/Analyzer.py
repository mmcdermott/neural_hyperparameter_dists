import numpy as np, scipy.stats as ss, pandas as pd

class ParamsMetrics():
    def __init__(
        self, accessor_metrics, fn_metrics
    ):
        self.metrics = fn_metrics
        acc = lambda x: lambda y: y[x]
        self.metrics.update({x: acc(x) for x in accessor_metrics})

    def measure(self, params_list, raw_perfs_list):
        best_perf, best_idx = -float('inf'), None
        valid_indices, perfs_list = [], []
        for i, perf in enumerate(raw_perfs_list):
            if np.isnan(perf): continue
            valid_indices.append(i)
            perfs_list.append(perf)
            if perf > best_perf: best_perf, best_idx = perf, i

        data = {
            'best_value': [], 'avg_value': [], 'avg_std': [], 'ttest_p': [],
            'correlation_coef': [], 'correlation_p': [],
        }
        metrics = []

        for metric, metric_fn in self.metrics.items():
            metrics.append(metric)

            vals = [metric_fn(p) for p in params_list]
            best = vals[best_idx]

            data['best_value'].append(best)

            compare_vals = [v for i, v in enumerate(vals) if not np.isnan(v) and i in valid_indices]
            compare_vals = np.array(compare_vals)

            # .dtype is np.dtype('bool')

            c_mean, c_std = np.mean(compare_vals), np.std(compare_vals)
            t_stat, p = ss.ttest_ind_from_stats(c_mean, c_std, len(compare_vals), best, 0, 1)
            data['avg_value'].append(c_mean)
            data['avg_std'].append(c_std)
            data['ttest_p'].append(p)

            corr_coef, corr_p = ss.spearmanr(compare_vals, perfs_list)
            data['correlation_coef'].append(corr_coef)
            data['correlation_p'].append(corr_p)

#             print(
#                 "%s prefers %.2e to %.2e ± %.2e (p=%.2e), "
#                 "and has correlation coefficient %.2f (p=%.2e) with overall performance"
#                 "" % (metric, best, c_mean, c_std, p, corr_coef, corr_p)
#             )

        return pd.DataFrame(data, index=metrics)

