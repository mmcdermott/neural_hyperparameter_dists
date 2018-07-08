import os, random, hashlib, pickle, numpy as np, pandas as pd

HYPERPARAMETERS_FILENAME = 'hyperparameters.pkl'
LOCAL_RESULTS_FILENAME   = 'results.pkl'
GLOBAL_RESULTS_FILENAME  = 'results.hdf'
PARAMS                   = 'Hyperparameters'
SPLIT                    = 'Split Number'

SPLIT_FN_TMPL            = 'split_%d'

def __sample_dict(d): return {k: v.rvs(1) for k, v in d.items()}

def sample(num_sample, hyperparameter_dists):
    if type(hyperparameter_dists) is dict: return __sample_dict(hyperparameter_dists)
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
    num_samples,
    seed,
    root_dir,
    global_results_fn = GLOBAL_RESULTS_FILENAME,
    hyperparameter_fn = HYPERPARAMETERS_FILENAME,
    local_results_fn  = LOCAL_RESULTS_FILENAME,
):
    np.random.seed(seed)
    random.seed(seed)

    index, data = [], {'train_score': [], 'local_score': [], 'held_out_score': []}
    results_filepath = os.path.join(root_dir, global_results_fn)

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

            try:
                M = model_cnstr(checkpoint_dir=split_dir, **hyperparameters) # TODO(mmd): filename?

                train, local_eval, held_out_eval =  dfs
                M.train(train)

                results = [M.evaluate(x) for x in dfs]
                # TODO(mmd): maybe make dictionaries?
                with open(os.path.join(split_dir, local_results_fn), mode='wb') as f: pickle.dump(results, f)
                train_result, local_result, held_out_result = results
            except Exception as e:
                # TODO(mmd): Store more information--stack trace...
                print("Exception!", num_sample, split_num, hyperparameters, e)

                with open(os.path.join(split_dir, 'errors.pkl'), mode='wb') as f: pickle.dump(e, f)

                train_result, local_result, held_out_result = np.NaN, np.NaN, np.NaN

            index.append((repr(sorted(hyperparameters.items())), split_num))
            data['train_score'].append(train_result)
            data['local_score'].append(local_result)
            data['held_out_score'].append(held_out_result)

        results_df, index, data = update_results_df(results_df, index, data)
