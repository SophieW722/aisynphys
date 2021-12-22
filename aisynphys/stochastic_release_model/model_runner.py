import os, time, pickle, logging, functools, multiprocessing
from collections import OrderedDict
import numpy as np
import sqlalchemy.orm
from .file_management import model_result_cache_path
from .model import StochasticReleaseModel


logger = logging.getLogger(__name__)


class StochasticModelRunner:
    """Handles loading data for a synapse and executing the model across a parameter space.
    """
    def __init__(self, db, experiment_id, pre_cell_id, post_cell_id, workers=None, load_cache=True, save_cache=False, cache_path=None):
        self.db = db
        self.experiment_id = experiment_id
        self.pre_cell_id = pre_cell_id
        self.post_cell_id = post_cell_id
        self.title = "%s %s %s" % (experiment_id, pre_cell_id, post_cell_id)
        
        self.workers = workers
        self.max_events = None
        
        self._synapse_events = None
        self._parameters = None
        self._param_space = None

        self._cache_path = cache_path or model_result_cache_path()
        self.cache_file = os.path.join(self._cache_path, "%s_%s_%s.pkl" % (experiment_id, pre_cell_id, post_cell_id))
        # whether to save cache file after running
        self._save_cache = save_cache

        if load_cache and os.path.exists(self.cache_file):
            self.load_result(self.cache_file)

    def run_model(self, params, **kwds):
        """Run the model for *params* and return a StochasticModelResult instance.
        """
        model = StochasticReleaseModel(params)
        spike_times, amplitudes, bg, event_meta = self.synapse_events
        if 'mini_amplitude' in params:
            return model.run_model(spike_times, amplitudes, event_meta=event_meta, **kwds)
        else:
            return model.optimize_mini_amplitude(spike_times, amplitudes, event_meta=event_meta, **kwds)

    @property
    def param_space(self):
        """A ParameterSpace instance containing the model output over the entire parameter space.
        """
        if self._param_space is None:
            self._param_space = self.generate_param_space()
            if self._save_cache:
                self.store_result(self.cache_file)
        return self._param_space

    def best_params(self):
        """Return a dict of parameters from the highest likelihood model run.
        """
        likelihood = self.param_space.result['likelihood']
        best_index = np.unravel_index(likelihood.argmax(), likelihood.shape)
        return self.param_space.params_at_index(best_index)

    def best_result(self):
        """Return the StochasticModelResult with the highest likelihood value.
        """
        return self.run_model(self.best_params())

    def generate_param_space(self):
        """Run the model across all points in the parameter space.
        """
        search_params = self.parameters
        
        param_space = ParameterSpace(search_params)

        # run once to jit-precompile before measuring preformance
        self.run_model(param_space.params_at_index((0,) * len(search_params)))

        start = time.time()
        import cProfile
        # prof = cProfile.Profile()
        # prof.enable()
        
        param_space.run(self.run_model, workers=self.workers)
        # prof.disable()
        logger.info("Run time: %f", time.time() - start)
        # prof.print_stats(sort='cumulative')
        
        return param_space

    def store_result(self, cache_file):
        path = os.path.dirname(cache_file)
        if not os.path.exists(path):
            os.makedirs(path)

        tmp = cache_file + '.tmp'
        pickle.dump(self.param_space, open(tmp, 'wb'))
        os.rename(tmp, cache_file)

    def load_result(self, cache_file):
        self._param_space = pickle.load(open(cache_file, 'rb'))
        
    @property
    def synapse_events(self):
        """Tuple containing (spike_times, amplitudes, baseline_amps, extra_info)
        """
        if self._synapse_events is None:
            session = self.db.session()
            try:
                self._synapse_events = self._load_synapse_events(session)
            finally:
                # For HPC, we have many processes and so can't use connection pooling effectively.
                # Instead, we need to be careful about closing connections when we're done with them.
                session.close()
        return self._synapse_events

    def get_pair(self, session=None):
        expt = self.db.experiment_from_ext_id(self.experiment_id, session=session)
        return expt.pairs[(self.pre_cell_id, self.post_cell_id)]

    def _load_synapse_events(self, session):
        pair = self.get_pair(session)
        syn_type = pair.synapse.synapse_type
        logger.info("Synapse type: %s", syn_type)

        # 1. Get a list of all presynaptic spike times and the amplitudes of postsynaptic responses

        events = self._event_query(pair, self.db, session).dataframe(rename_columns=False)
        logger.info("loaded %d events", len(events))

        if len(events) == 0:
            raise Exception("No events found for this synapse.")

        rec_times = (events['rec_start_time'] - events['rec_start_time'].iloc[0]).dt.total_seconds().to_numpy()
        spike_times = events['first_spike_time'].to_numpy().astype(float) + rec_times
        
        # some metadata to follow the events around--not needed for the model, but useful for 
        # analysis later on.
        event_meta = events[['sync_rec_ext_id', 'pulse_number', 'induction_frequency', 'recovery_delay', 'stim_name']]
        
        # any missing spike times get filled in with the average latency
        missing_spike_mask = np.isnan(spike_times)
        logger.info("%d events missing spike times", missing_spike_mask.sum())
        avg_spike_latency = np.nanmedian(events['first_spike_time'] - events['onset_time'])
        pulse_times = events['onset_time'] + avg_spike_latency + rec_times
        spike_times[missing_spike_mask] = pulse_times[missing_spike_mask]

        # get individual event amplitudes
        amplitudes = events['dec_fit_reconv_amp'].to_numpy().astype(float)
        if np.isfinite(amplitudes).sum() == 0:
            raise Exception("No event amplitudes available for this synapse.")
        
        # filter events by inhibitory or excitatory qc
        qc_field = syn_type + '_qc_pass'
        qc_mask = events[qc_field] == True
        logger.info("%d events passed qc", qc_mask.sum())
        amplitudes[~qc_mask] = np.nan
        amplitudes[missing_spike_mask] = np.nan
        logger.info("%d good events to be analyzed", np.isfinite(amplitudes).sum())
        
        # get background events for determining measurement noise
        bg_amplitudes = events['baseline_dec_fit_reconv_amp'].to_numpy().astype(float)
        # filter by qc
        bg_qc_mask = events['baseline_'+qc_field] == True
        bg_amplitudes[~qc_mask] = np.nan
        
        # first_pulse_mask = events['pulse_number'] == 1
        # first_pulse_amps = amplitudes[first_pulse_mask]
        # first_pulse_stdev = np.nanstd(first_pulse_amps)
        # first_pulse_mean = np.nanmean(first_pulse_amps)
        
        if self.max_events is not None:
            spike_times = spike_times[:self.max_events]
            amplitudes = amplitudes[:self.max_events]
        
        return spike_times, amplitudes, bg_amplitudes, event_meta

    @staticmethod
    def _event_query(pair, db, session):
        pre_rec = sqlalchemy.orm.aliased(db.Recording, name='pre_rec')

        q = session.query(
            db.PulseResponse,
            db.PulseResponse.ex_qc_pass,
            db.PulseResponse.in_qc_pass,
            db.Baseline.ex_qc_pass.label('baseline_ex_qc_pass'),
            db.Baseline.in_qc_pass.label('baseline_in_qc_pass'),
            db.PulseResponseFit.fit_amp,
            db.PulseResponseFit.dec_fit_reconv_amp,
            db.PulseResponseFit.fit_nrmse,
            db.PulseResponseFit.baseline_fit_amp,
            db.PulseResponseFit.baseline_dec_fit_reconv_amp,
            db.StimPulse.first_spike_time,
            db.StimPulse.pulse_number,
            db.StimPulse.onset_time,
            pre_rec.stim_name,
            db.Recording.start_time.label('rec_start_time'),
            db.PatchClampRecording.baseline_current,
            db.MultiPatchProbe.induction_frequency,
            db.MultiPatchProbe.recovery_delay,
            db.SyncRec.ext_id.label('sync_rec_ext_id'),
        )
        q = q.join(db.Baseline, db.PulseResponse.baseline)
        q = q.join(db.PulseResponseFit, isouter=True)
        q = q.join(db.StimPulse)
        q = q.join(pre_rec, pre_rec.id==db.StimPulse.recording_id)
        q = q.join(db.Recording, db.PulseResponse.recording)
        q = q.join(db.SyncRec, db.Recording.sync_rec)
        q = q.join(db.PatchClampRecording)
        q = q.join(db.MultiPatchProbe, isouter=True)

        q = q.filter(db.PulseResponse.pair_id==pair.id)
        q = q.filter(db.PatchClampRecording.clamp_mode=='ic')

        q = q.order_by(db.Recording.start_time).order_by(db.StimPulse.onset_time)

        return q

    @property
    def parameters(self):
        """A structure defining the parameters to search.
        """
        if self._parameters is None:
            self._parameters = self._generate_parameters()
        return self._parameters  

    def _generate_parameters(self):
        spike_times, amplitudes, bg_amplitudes, event_meta = self.synapse_events
        
        measurement_stdev = np.nanstd(bg_amplitudes)
        assert np.isfinite(measurement_stdev), "Could not measure background amplitude stdev (no finite values available)"

        def logspace(start, finish, n):
            return start * ((finish / start)**(1 / (n-1)))**np.arange(n)

        search_params = {
            # If mini_amplitude is commented out here, then it will be optimized automatically by the model:
            #'mini_amplitude': np.nanmean(amplitudes) * 1.2**np.arange(-12, 24, 2),

            'n_release_sites': np.array([1, 2, 3, 4, 6, 8, 12, 16, 24, 32]),
            # 'n_release_sites': np.array([1, 2, 4, 8, 16, 32]),
            'base_release_probability': logspace(0.01, 1.0, 18),
            'mini_amplitude_cv': logspace(0.1, 1.0, 6),
            'measurement_stdev': measurement_stdev,

            'depression_amount': np.array([-1] + list(np.linspace(0.0, 1.0, 9))),
            'depression_tau': logspace(0.01, 2.56, 8),
            # 'depression_amount': np.array([-1, 0.0, 0.1, 0.3, 0.6, 1.0]),
            # 'depression_tau': np.array([0.01, 0.04, 0.16, 0.64, 2.56]),
            # 'depression_amount': np.array([-1, 0, 0.5]),
            # 'depression_tau': np.array([0.0001, 0.01]),

            'facilitation_amount': np.linspace(0.0, 1.0, 9),
            'facilitation_tau': logspace(0.01, 2.56, 8),
            # 'facilitation_amount': np.array([0.0, 0.1, 0.3, 0.6, 1.0]),
            # 'facilitation_tau': np.array([0.01, 0.04, 0.16, 0.64, 2.56]),
            # 'facilitation_amount': np.array([0, 0.5]),
            # 'facilitation_tau': np.array([0.1, 0.5]),

        }
        
        # sanity checking
        for k,v in search_params.items():
            if np.isscalar(v):
                assert not np.isnan(v), k
            else:
                assert not np.any(np.isnan(v)), k

        print("Parameter space:")
        for k, v in search_params.items():
            print("   ", k, v)

        size = np.product([(len(v) if not np.isscalar(v) else 1) for v in search_params.values()])
        print("Total size: %d  (%0.2f MB)" % (size, size*8*1e-6))

        return search_params


class CombinedModelRunner:
    """Model runner combining the results from multiple StochasticModelRunner instances.
    """
    def __init__(self, runners):
        self.model_runners = runners
        self.title = " : ".join(r.title for r in runners)
        
        params = OrderedDict()
        # params['synapse'] = [
        #     '%s_%s_%s' % (args.experiment_id, args.pre_cell_id, args.post_cell_id),
        #     '%s_%s_%s' % (args.experiment_id2, args.pre_cell_id2, args.post_cell_id2),
        # ]
        params['synapse'] = np.arange(len(runners))
        params.update(runners[0].param_space.params)
        param_space = ParameterSpace(params)
        param_space.result = np.stack([runner.param_space.result for runner in runners])
        
        self.param_space = param_space
        
    def run_model(self, params):
        params = params.copy()
        runner = self.model_runners[params.pop('synapse')]
        return runner.run_model(params)


class ParameterSpace(object):
    """Used to generate and store model results over a multidimentional parameter space.

    Parameters
    ----------
    params : dict
        Dictionary of {'parameter_name': array_of_values} describing parameter space to be searched.
        Scalar values are passed to the evaluation function when calling run(), but are not included
        as an axis in the result array.

    """
    def __init__(self, params):
        self.params = params
        
        static_params = {}
        for param, val in list(params.items()):
            if np.isscalar(val):
                static_params[param] = params.pop(param)
        self.static_params = static_params
        self.param_order = list(params.keys())
        self.shape = tuple([len(self.params[p]) for p in self.param_order])
        self.result = None
        
    def axes(self):
        """Return an ordered dictionary giving the axis (parameter) names and the parameter values along each axis.
        """
        return OrderedDict([(ax, {'values': self.params[ax]}) for ax in self.param_order])

    def run(self, func, workers=None, **kwds):
        """Run *func* in parallel over the entire parameter space, storing
        results into self.result.

        If workers==1, then run locally to make debugging easier.
        """
        if workers is None:
            workers = multiprocessing.cpu_count()
        all_inds = list(np.ndindex(self.shape))
        all_params = [self.params_at_index(inds) for inds in all_inds]

        example_result = func(all_params[0], **kwds)
        opt_keys = list(example_result.optimized_params.keys())
        dtype = [(k, 'float32') for k in ['likelihood'] + opt_keys]
        self.result = np.empty(self.shape, dtype=dtype)

        if workers > 1:
            # multiprocessing can be flaky.. if pool.imap or pool.terminate never return,
            # try switching between 'fork' and 'spawn':
            ctx = multiprocessing.get_context('spawn')            
            pool = ctx.Pool(workers)
            
            fn = functools.partial(func, **kwds)

            from aisynphys.ui.progressbar import ProgressBar
            with ProgressBar(f'synapticulating ({workers} workers)...', maximum=len(all_inds)) as dlg:
                for i,r in enumerate(pool.imap(fn, all_params, chunksize=100)):
                    try:
                        dlg.update(i)
                    except dlg.CanceledError:
                        pool.terminate()
                        raise Exception("Synapticulation cancelled. No refunds.")
                    self.result[all_inds[i]] = (r.likelihood,) + tuple([r.optimized_params[k] for k in opt_keys])
        else:
            from aisynphys.ui.progressbar import ProgressBar
            with ProgressBar('synapticulating... (serial)', maximum=len(all_inds)) as dlg:
                for i,inds in enumerate(all_inds):
                    params = self.params_at_index(inds)
                    r = func(params, **kwds)
                    self.result[inds] = (r.likelihood,) + tuple([r.optimized_params[k] for k in opt_keys])
                    try:
                        dlg.update(i)
                    except dlg.CanceledError:
                        raise Exception("Synapticulation cancelled. No refunds.")
        
    def params_at_index(self, inds):
        """Return a dict of the parameter values used at a specific tuple index in the parameter space.
        """
        params = self.static_params.copy()
        for i,param in enumerate(self.param_order):
            params[param] = self.params[param][inds[i]]
        return params

    def index_at_params(self, params):
        """Return a tuple of indices giving the paramete space location of the given parameter values.
        """
        axes = self.axes()
        inds = []
        for pname in enumerate(self.param_order):
            i = np.argwhere(axes[pname]==params[pname])[0,0]
            inds.append(i)
        return tuple(inds)
