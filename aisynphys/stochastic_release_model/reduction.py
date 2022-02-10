import os, pickle, gc, time, traceback, functools
import numpy as np
import sklearn.preprocessing, sklearn.decomposition
from .file_management import list_cached_results, load_cached_model_results
from aisynphys.database import default_db as db
from aisynphys import config


def default_spca_file(likelihood_only=False):
    """Return default file name for sparse PCA results.
    """
    run_type = 'likelihood_only' if likelihood_only else 'all_results'
    return config.stochastic_model_spca_file.format(run_type=run_type)


def reduce_model_results(output_file=None, likelihood_only=False, cache_path=None):
    """Load all cached result files from the stochastic release model, concatenate into 
    a single array, and reduce using sparse PCA.

    This function requires a large amount of memory (1TB for original coarse matrix data) and CPU time.

    Parameters
    ----------
    output_file : str | None
        File to store SPCA results in. If None, then the filename is derived from aisynphys.config.release_model_spca_file
    likelihood_only : bool
        If True, then calculate SPCA based only on the likelihood values output from the model.
        If False, then use likelihood values as well as any fit parameters (probably mini_amplitude)
    cache_path : str | None
        Path where cached model files can be found. If None, then the default path is used (see
        aisynphys.stochastic_release_model.model_result_cache_path)
    """
    if output_file is None:
        output_file = default_spca_file(likelihood_only)
    cache_files = [result[1] for result in list_cached_results(cache_path)][:10]
    print(f"Generating SPCA reduction from {len(cache_files)} cached model files; writing to {output_file}")

    ## Load all model outputs into a single array
    agg_result, cache_files, param_space = load_cached_model_results(cache_files, db=db)
    agg_shape = agg_result.shape
    print("  cache loaded.")

    if likelihood_only:
        # use only mode likelihood; no amplitude
        flat_result = agg_result[...,0].reshape(agg_shape[0], np.product(agg_shape[1:-1]))
    else:
        # use complete result array
        flat_result = agg_result.reshape(agg_shape[0], np.product(agg_shape[1:]))
    print("  reshape.")

    # Prescale model data
    print("   Fitting prescaler...")
    scaler = sklearn.preprocessing.StandardScaler()
    n_obs = flat_result.shape[0]
    chunk_size = 10
    n_chunks = n_obs // chunk_size

    scaler.fit(flat_result)
    print("   Prescaler transform...")
    scaled = scaler.transform(flat_result)

    print("   Prescaler done.")

    # free up some memory
    del agg_result
    del flat_result
    gc.collect()

    print("free memory")

    # fit sparse PCA  (uses ~6x memory of input data)
    try:
        start = time.time()
        print("Fitting sparse PCA...")
        n_pca_components = 50
        pca = sklearn.decomposition.MiniBatchSparsePCA(n_components=n_pca_components, n_jobs=-1)
        pca.fit(scaled)
        print("  Sparse PCA fit complete.")
   
        # run sparse PCA
        print("Sparse PCA transform...")
        sparse_pca_result = pca.transform(scaled)
        pickle.dump({
            'result': sparse_pca_result, 
            'params': param_space, 
            'cache_files': cache_files, 
            'sparse_pca': pca, 
            'scaler': scaler,
        }, open(output_file, 'wb'))
        print("   Sparse PCA transform complete: %s" % output_file)
    except Exception as exc:
        print("Sparse PCA failed:")
        traceback.print_exc()
    finally:
        print("Sparse PCA time: %d sec" % int(time.time()-start))


# fit standard PCA   (uses ~2x memory of input data)
#try:
#    start = time.time()
#    print("Fitting PCA...")
#    n_pca_components = 30#500
#    pca = sklearn.decomposition.PCA(n_components=n_pca_components)
#    pca.fit(scaled)
#    print("  PCA fit complete.")
#
#    # run PCA
#    print("PCA transform...")
#    pca_result = pca.transform(scaled)
#    pca_file = os.path.join(cache_path, 'pca.pkl')
#    pickle.dump({'result': pca_result, 'params': param_space, 'cache_files': cache_files, 'pca': pca}, open(pca_file, 'wb'))
#    print("   PCA transform complete: %s" % pca_file)
#except Exception as exc:
#    print("PCA failed:")
#    traceback.print_exc()
#finally:
#    print("PCA time: %d sec" % int(time.time()-start))


# umap  (uses ~1x memory of input data)
#try:
#    start = time.time()
#    n_umap_components = 15#32
#    reducer = umap.UMAP(
#        n_components=n_umap_components,
#    #     n_neighbors=5,
#        low_memory=False,
#        init='spectral',   # also try 'random'
#        verbose=True,
#    )
#
#    print("Fit UMAP...")
#    reducer.fit(scaled)
#    print("   UMAP fit done.")
#
#    print("UMAP transform...")
#    umap_result = reducer.transform(scaled)
#    umap_file = os.path.join(cache_path, 'umap.pkl')
#    pickle.dump({'result': umap_result, 'params': param_space, 'cache_files': cache_files}, open(umap_file, 'wb'))
#    pickle.dump(reducer, open('umap_model.pkl', 'wb'))
#    print("   UMAP transform complete: %s" % umap_file)
#except Exception as exc:
#    print("UMAP failed:")
#    traceback.print_exc()
#finally:
#    print("UMAP time: %d sec" % int(time.time()-start))




@functools.lru_cache(maxsize=1)
def load_spca_results(max_vector_size=None, likelihood_only=False):
    """Return a dict containing results of sparse PCA run on the posterior distribution of likelihood
    values across model parameters.

    Results are originally generated by aisynphys/analyses/stochastic_model_reduction.py and stored
    in the location specified by config.release_model_spca_file.

    Contains:
    - model: sklearn sparse PCA model
    - param_space: release model

    """
    spca_file = default_spca_file(likelihood_only)

    sm_results = pickle.load(open(spca_file, 'rb'))

    results = {
        'model': sm_results['sparse_pca'],
        'param_space': sm_results['params'],
        'synapse_vectors': {},
    }
    for i, cache_file in enumerate(sm_results['cache_files']):
        expt_id, pre_cell_id, post_cell_id = os.path.split(os.path.splitext(cache_file)[0])[1].split('_')
        syn_id = (expt_id, pre_cell_id, post_cell_id)
        vector = sm_results['result'][i]
        if max_vector_size is not None:
            vector = vector[:max_vector_size]
        results['synapse_vectors'][syn_id] = vector

    return results
