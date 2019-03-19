import os, random, hashlib, pickle, numpy as np, pandas as pd

from .distributions import DictDistribution

HYPERPARAMETERS_FILENAME = 'hyperparameters.pkl'
LOCAL_RESULTS_FILENAME   = 'results.pkl'
GLOBAL_RESULTS_FILENAME  = 'results.hdf'
PARAMS                   = 'Hyperparameters'
SPLIT                    = 'Split Number'

SPLIT_FN_TMPL            = 'split_%d'

# TODO(mmd): Add 'split respects'
def make_cv_indices(
    pd_idx,
    K,
    return_integral_indices=False,
    shuffle=True,
):
    """
    Returns a list of K folds, each a tuple of (train_idx, tuning_eval_idx, held_out_eval_idx), where idx
    is either directly in pd_idx space or in integral_idx space if return_integral_indices
    """

    N = len(pd_idx)

    base_idx = np.arange(N) if return_integral_indices else pd_idx.values
    if shuffle: base_idx = np.random.permutation(base_idx)

    fold_indices = np.array_split(base_idx, K)
    CV_splits = []
    for i in range(K):
        tuning_eval_fold = i
        held_out_eval_fold = (i+1) % K
        train_folds = [j for j in range(K) if j not in [tuning_eval_fold, held_out_eval_fold]]
        CV_splits.append((
            np.concatenate([fold_indices[j] for j in train_folds]), 
            fold_indices[tuning_eval_fold],
            fold_indices[held_out_eval_fold]
        ))

    return CV_splits

def sample(num_sample, hyperparameter_dists):
    if type(hyperparameter_dists) is dict: return DictDistribution(hyperparameter_dists).rvs(1)[0]
    elif type(hyperparameter_dists) in [list, tuple]: return hyperparameter_dists[num_sample]

    # TODO(mmd): copy from other work
    raise NotImplementedError

def __hash_dict(d):
    H = hashlib.new('md4')
    H.update(repr(sorted(d.items())).encode('utf-8'))
    return H.hexdigest()

# TODO(mmd): Function to back out results df from split directories.

def run_CV(
    CV_splits,
    hyperparameter_dists,
    model_cnstr,
    evaluators,
    num_samples,
    seed,
    root_dir,
    global_results_fn = GLOBAL_RESULTS_FILENAME,
    hyperparameter_fn = HYPERPARAMETERS_FILENAME,
    local_results_fn  = LOCAL_RESULTS_FILENAME,
):
    np.random.seed(seed)
    random.seed(seed)

    if type(evaluators) is not dict: evaluators = {'eval': evaluators}

    index, data = [], {'train_score': [], 'local_score': [], 'held_out_score': []}
    results_filepath = os.path.join(root_dir, global_results_fn)
    assert results_filepath.endswith('.hdf'), "Only supports h5 files at this stage."

    def update_results_df(old_results_df, index, data):
        new_results_df = pd.DataFrame(data, index=pd.MultiIndex.from_tuples(index, names=[PARAMS, SPLIT]))
        index, data = [], {'train_score': [], 'local_score': [], 'held_out_score': []}

        if old_results_df is None: results_df = new_results_df
        else: results_df = pd.concat([old_results_df, new_results_df])

        results_df.to_hdf(results_filepath, 'all')
        return results_df, index, data

    if os.path.isfile(results_filepath): results_df = pd.read_hdf(results_filepath)
    else: results_df, index, data = update_results_df(None, index, data)

    for num_sample in range(num_samples):
        hyperparameters = sample(num_sample, hyperparameter_dists)

        setting_dir = os.path.join(root_dir, __hash_dict(hyperparameters))
        assert not os.path.isdir(setting_dir), "Hyperparameters already tried! %s" % str(hyperparameters)
        os.mkdir(setting_dir)

        with open(os.path.join(setting_dir, hyperparameter_fn), mode='wb') as f:
            # TODO(mmd): How to deal with functions?
            pickle.dump(hyperparameters, f)

        for split_num, dfs in enumerate(CV_splits):
            print('Starting on split %d' % split_num)

            split_dir = os.path.join(setting_dir, SPLIT_FN_TMPL % split_num)
            if not os.path.isdir(split_dir): os.mkdir(split_dir)

            try:
                M = model_cnstr(checkpoint_dir=split_dir, **hyperparameters) # TODO(mmd): filename?

                train, local_eval, held_out_eval =  dfs

                if type(train) is tuple: M.fit(*train)
                elif type(train) is dict: M.fit(**train)
                else: M.fit(train)

                results = {
                    eval_name: [
                        (
                            evaluator.score(model=M, **x) if type(x) is dict else evaluator.score(M, x)
                        ) for x in dfs
                    ] for eval_name, evaluator in evaluators.items()
                }
                # TODO(mmd): maybe make dictionaries?
                with open(os.path.join(split_dir, local_results_fn), mode='wb') as f: pickle.dump(results, f)
                train_result, local_result, held_out_result = results
            except Exception as e:
                # TODO(mmd): Store more information--stack trace...
                print("Exception!", num_sample, split_num, hyperparameters, e)
                raise

                with open(os.path.join(split_dir, 'errors.pkl'), mode='wb') as f: pickle.dump(e, f)

                train_result, local_result, held_out_result = np.NaN, np.NaN, np.NaN

            index.append((repr(sorted(hyperparameters.items())), split_num))
            data['train_score'].append(train_result)
            data['local_score'].append(local_result)
            data['held_out_score'].append(held_out_result)

        results_df, index, data = update_results_df(results_df, index, data)
