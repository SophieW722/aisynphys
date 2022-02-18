"""
Script used for collecting stochastic release model results and running dimensionality reduction on them.

Depending on the number of synapses to process and the size of the model parameter space, this
may consume a large amount of memory (~1TB for original coarse matrix data) and CPU time.

"""
import argparse
from aisynphys import config
from aisynphys.stochastic_release_model.reduction import reduce_model_results


parser = argparse.ArgumentParser(parents=[config.parser])
parser.add_argument('--output-file', type=str, default=None, dest='output_file', help="Optional file name to write")
parser.add_argument('--cache-path', type=str, default=None, dest='cache_path', help="Optional path to model cache files")
args = parser.parse_args()


for likelihood_only in (False, True):
    # use default cache path and output file
    reduce_model_results(likelihood_only=likelihood_only, cache_path=args.cache_path, output_file=args.output_file)

