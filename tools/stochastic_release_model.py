import os, sys, argparse, threading, time, subprocess
from aisynphys.database import default_db as db
from aisynphys.stochastic_release_model import StochasticModelRunner, CombinedModelRunner
from aisynphys import config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[config.parser])
    parser.add_argument('experiment_id', type=str, nargs='?')
    parser.add_argument('pre_cell_id', type=str, nargs='?')
    parser.add_argument('post_cell_id', type=str, nargs='?')
    parser.add_argument('experiment_id2', type=str, nargs='?')
    parser.add_argument('pre_cell_id2', type=str, nargs='?')
    parser.add_argument('post_cell_id2', type=str, nargs='?')
    parser.add_argument('--workers', type=int, default=None, help="The number of parallel processes to use")
    parser.add_argument('--max-events', type=int, default=None, dest='max_events', help="Limit the numper of events to be processed (for testing)")
    parser.add_argument('--no-cache', default=False, action='store_true', dest='no_cache', help="Ignore existing cached files")
    parser.add_argument('--no-gui', default=False, action='store_true', dest='no_gui', help="Disable GUI; just generate cache results")
    parser.add_argument('--cache-path', type=str, default=None, dest='cache_path', help="Path for reading/writing cache files")
    
    args = parser.parse_args()

    if not args.no_gui:
        import pyqtgraph as pg
        app = pg.mkQApp()
        if sys.flags.interactive == 1:
            pg.dbg()

    def load_experiment(experiment_id, pre_cell_id, post_cell_id):
        print("Loading stochastic model for %s %s %s" % (experiment_id, pre_cell_id, post_cell_id))

        result = StochasticModelRunner(db, experiment_id, pre_cell_id, post_cell_id, workers=args.workers)
        result.max_events = args.max_events
        cache_path = args.cache_path or os.path.join(config.cache_path, 'stochastic_model_results')
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        cache_file = os.path.join(cache_path, "%s_%s_%s.pkl" % (experiment_id, pre_cell_id, post_cell_id))
        if not args.no_cache and os.path.exists(cache_file):
            result.load_result(cache_file)
        else:
            print("cache miss:", cache_file)
            result.store_result(cache_file)
        return result

    # load 1 or 2 experiments:        
    result1 = load_experiment(args.experiment_id, args.pre_cell_id, args.post_cell_id)
    result2 = None if args.experiment_id2 is None else load_experiment(args.experiment_id2, args.pre_cell_id2, args.post_cell_id2)

    result = result1 if result2 is None else CombinedModelRunner([result1, result2])
    
    # 4. Visualize parameter space.

    if not args.no_gui:
        from aisynphys.ui.stochastic_release_model import ModelDisplayWidget
        win = ModelDisplayWidget(result)
        win.setWindowTitle(result.title)
        win.show()

        if sys.flags.interactive == 0:
            app.exec_()
