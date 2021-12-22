"""
Script used for collecting stochastic release model results and running dimensionality reduction on them.

Depending on the number of synapses to process and the size of the model parameter space, this
may consume a large amount of memory (~1TB for original coarse matrix data) and CPU time.

"""

from aisynphys.stochastic_release_model.reduction import reduce_model_results


for likelihood_only in (False, True):
    # use default cache path and output file
    reduce_model_results(likelihood_only=likelihood_only)

