import os, random, hashlib, pickle, numpy as np, pandas as pd

HYPERPARAMETERS_FILENAME = 'hyperparameters.pkl'
LOCAL_RESULTS_FILENAME   = 'results.pkl'
GLOBAL_RESULTS_FILENAME  = 'results.hdf'
PARAMS                   = 'Hyperparameters'
SPLIT                    = 'Split Number'

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
    if os.path.isfile(results_filepath): results_df = pd.read_hdf(results_filepath)
    else: results_df = pd.DataFrame(data, index=pd.MultiIndex.from_tuples(index, names=[PARAMS, SPLIT]))

    for num_sample in range(num_samples):
        hyperparameters = sample(num_sample, hyperparameter_dists)

        setting_dir = os.path.join(root_dir, __hash_dict(hyperparameters))
        assert not os.path.isdir(setting_dir), "Hyperparameters already tried! %s" % str(hyperparameters)
        os.mkdir(setting_dir)

        with open(os.path.join(setting_dir, hyperparameter_fn), mode='wb') as f:
            pickle.dump(hyperparameters, f)

        for split_num, dfs in enumerate(CV_splits):
            print('Starting on split %d' % split_num)

            split_dir = os.path.join(setting_dir, 'split_%d' % split_num)

            M = model_cnstr(checkpoint_dir=split_dir, **hyperparameters) # TODO(mmd): filename?

            train, local_eval, held_out_eval =  dfs
            M.train(train)

            results = [M.evaluate(x) for x in dfs]
            # TODO(mmd): maybe make dictionaries?
            with open(os.path.join(split_dir, local_results_fn), mode='wb') as f: pickle.dump(results, f)
            train_result, local_result, held_out_result = results

            index.append((hyperparameters, split_num))
            data['train_score'].append(train_result)
            data['local_score'].append(local_result)
            data['held_out_score'].append(held_out_result)

    results_df = pd.concat([results_df, pd.DataFrame(data, index=pd.MultiIndex.from_tuples(index,
        names=[PARAMS, SPLIT]))])
    resutls_df.to_hdf(results_filepath
