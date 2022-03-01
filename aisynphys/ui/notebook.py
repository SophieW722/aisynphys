"""Commonly used routines when working with jupyter / matplotlib
"""
import io
import numpy as np
import matplotlib
import matplotlib.cm
import matplotlib.lines
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from textwrap import wrap
from scipy import stats
from aisynphys.cell_class import CellClass
from neuroanalysis.data import TSeries
from neuroanalysis.baseline import float_mode
from aisynphys.avg_response_fit import response_query, sort_responses
from aisynphys.connectivity import connectivity_profile, distance_adjusted_connectivity
from aisynphys.data import PulseResponseList
from aisynphys.dynamics import stim_sorted_pulse_amp
from aisynphys.database import default_db as db


def heatmap(data, row_labels, col_labels, ax=None, ax_labels=None, bg_color=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.
    
    Modified from https://matplotlib.org/3.1.0/gallery/images_contours_and_fields/image_annotated_heatmap.html

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    ax_labels
        (x, y) axis labels
    bg_color
        Background color shown behind transparent pixels
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    if bg_color is not None:
        bg = np.empty(data.shape[:2] + (3,))
        bg[:] = matplotlib.colors.to_rgb(bg_color)        
        ax.imshow(bg)
        
    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)

    if ax_labels is not None:
        ax.set_ylabel(ax_labels[1], size=16)
        ax.set_xlabel(ax_labels[0], size=16)
        ax.xaxis.set_label_position('top')
    
    return im, cbar


def annotate_heatmap(im, labels, data=None, textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Modified from https://matplotlib.org/3.1.0/gallery/images_contours_and_fields/image_annotated_heatmap.html

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    labels
        Array of strings to display in each cell
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """
    pixels, _, _, _ = im.make_image(renderer=None, unsampled=True)

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            px_color = pixels[i,j]
            if isinstance(px_color, np.ma.core.MaskedArray):
                px_color = px_color.data
            kw['color'] =  textcolors[int(np.mean(px_color[:3]) < 128)]
            text = im.axes.text(j, i, labels[i, j], **kw)
            texts.append(text)

    return texts


def show_connectivity_matrix(ax, results, pre_cell_classes, post_cell_classes, class_labels, cmap, norm, ctype='chemical',
                             distance_adjusted=False, cbarlabel='Connection Probability', layer_lines=None, alpha=True,
                             corrections_applied=False, show_pmax=False, correction_only=False):
    """Display a connectivity matrix.

    This function uses matplotlib to display a heatmap representation of the output generated by
    aisynphys.connectivity.measure_connectivity(). Each element in the matrix is colored by connection 
    probability, and the connection probability confidence interval is used to set the transparency 
    such that the elements with the most data (and smallest confidence intervals) will appear
    in more bold colors. 

    Parameters
    ----------
    ax : matplotlib.axes
        The matplotlib axes object on which to draw the connectivity matrix
    results : dict
        Output of aisynphys.connectivity.measure_connectivity. This structure maps
        (pre_class, post_class) onto the results of the connectivity analysis.
    pre_cell_classes : list
        List of presynaptic cell classes in the order they should be displayed
    post_cell_classes : list
        List of postsynaptic cell classes in the order they should be displayed
    class_labels : dict
        Maps {cell_class: label} to give the strings to display for each cell class.
    cmap : matplotlib colormap instance
        The colormap used to generate colors for each matrix element
    norm : matplotlib normalize instance
        Normalize instance used to normalize connection probability values before color mapping
    ctype: string
        'chemical' or 'electrical'
    distance_adjusted: bool
        If True, use distance-adjusted connectivity metrics. See 
        ``aisynphys.connectivity.measure_connectivity(sigma)``.
    cbarlabel: string
        label for color bar
    alpha : float
        If True, apply transparency based on width of confidence intervals
    layer_lines : list
        List of integers to draw boundaries at. Assumes symmetrical matrix.
    corrections_applied: bool
        If True, try to read corrected version of the fit.
    show_pmax: bool
        If True, show the value of (estimated) probability in the matrix
    correction_only: bool
        If True, show the ratio between corrections-applied p_max vs distance-adjusted p_max
        Use with corrections_applied=True
    """
    # convert dictionary of results to a 2d array of connection probabilities
    shape = (len(pre_cell_classes), len(post_cell_classes))
    cprob = np.zeros(shape)
    cprob_alpha = np.zeros(shape)
    cprob_str = np.zeros(shape, dtype=object)

    for i,pre_class in enumerate(pre_cell_classes):
        for j,post_class in enumerate(post_cell_classes):
            result = results[pre_class, post_class]
            if ctype == 'chemical':
                found = result['n_connected']
                if distance_adjusted:
                    cp, cp_lower_ci, cp_upper_ci = result['adjusted_connectivity']
                elif corrections_applied:
                    cp, cp_lower_ci, cp_upper_ci = result['connectivity_correction_fit'].cp_ci
                    if correction_only:
                        cp = cp / result['connection_probability'][0] if result['connection_probability'][0] != 0 else np.nan
                        cp_lower_ci = cp # disabling the ci
                        cp_upper_ci = cp
                else:
                    cp, cp_lower_ci, cp_upper_ci = result['connection_probability']
            elif ctype == 'electrical':
                found = result['n_gaps']
                if distance_adjusted:
                    cp, cp_lower_ci, cp_upper_ci = result['adjusted_gap_junction']
                else:
                    cp, cp_lower_ci, cp_upper_ci = result['gap_probability']
            else:
                raise Exception('ctype must be one of "chemical" or "electrical"')
            
            if not correction_only:
                cp = min(cp, 1)
                cp_upper_ci = min(1, cp_upper_ci)
                cp_lower_ci = max(0, cp_lower_ci)

            cprob[i,j] = cp
            if ctype == 'chemical':
                cprob_str[i,j] = "" if result['n_probed'] == 0 else "%d/%d" % (found, result['n_probed'])
                if (show_pmax or correction_only) and np.isfinite(cp):
                    cprob_str[i,j] += "\n %.3f" %(cp)
            elif ctype == 'electrical':
                cprob_str[i,j] = "" if result['n_gaps_probed'] == 0 else "%d/%d" % (found, result['n_gaps_probed'])

            
            cprob_alpha[i,j] = 1.0 - 1.5 * max(cp_upper_ci - cp, cp - cp_lower_ci)


    # map connection probability to RGB colors
    mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cprob_rgba = mapper.to_rgba(np.clip(cprob, norm.vmin, norm.vmax))

    # apply alpha based on confidence intervals
    if alpha:
        cprob_rgba[:, :, 3] = np.clip(cprob_alpha, 0, 1)

    # generate lists of labels to display along the pre- and postsynaptic axes
    pre_class_labels = [class_labels[cls] for cls in pre_cell_classes]
    post_class_labels = [class_labels[cls] for cls in post_cell_classes]

    # draw the heatmap with axis labels and colorbar
    im, cbar = heatmap(cprob_rgba, pre_class_labels, post_class_labels, ax=ax, 
        ax_labels=('postsynaptic', 'presynaptic'),
        bg_color=(0.7, 0.7, 0.7),
        cmap=cmap, norm=norm, 
        cbarlabel=cbarlabel, 
        cbar_kw={'shrink':0.5})

    # draw text over each matrix element
    labels = annotate_heatmap(im, cprob_str, data=cprob)

    if layer_lines is not None:
        [ax.axhline(pos, color='white', linewidth=2) for pos in layer_lines]
        [ax.axvline(pos, color='white', linewidth=2) for pos in layer_lines]
    
    return im, cbar, labels


def generate_connectivity_matrix(db, cell_classes, pair_query_args, ax):
    from ..connectivity import measure_connectivity
    from ..cell_class import classify_cells, classify_pairs

    pairs = db.pair_query(**pair_query_args).all()

    # Group all cells by selected classes
    cell_groups = classify_cells(cell_classes.values(), pairs=pairs)

    # Group pairs into (pre_class, post_class) groups
    pair_groups = classify_pairs(pairs, cell_groups)

    # analyze matrix elements
    results = measure_connectivity(pair_groups, sigma=100e-6, dist_measure='lateral_distance')

    # define a colormap and log normalization used to color the heatmap
    norm = matplotlib.colors.LogNorm(vmin=0.01, vmax=1.0, clip=True)
    cmap = matplotlib.cm.get_cmap('plasma')    

    class_labels = {cls:name for name,cls in cell_classes.items()}

    # finally, draw the colormap using the provided function:
    im, cbar, labels = show_connectivity_matrix(
        ax=ax, 
        results=results, 
        pre_cell_classes=cell_classes.values(), 
        post_cell_classes=cell_classes.values(), 
        class_labels=class_labels, 
        cmap=cmap, 
        norm=norm,
        distance_adjusted=True
    )

    return results, (im, cbar, labels)


def get_metric_data(metric, db, pre_classes=None, post_classes=None, pair_query_args=None, metrics=None):
    synapse_metrics = {
        #                                     name                                  unit   scale alpha  db columns                                         colormap       log     clim           text format

        'psp_amplitude':                      ('PSP Amplitude',                     'mV',  1e3,  1,     [db.Synapse.psp_amplitude],                        'bwr',         False,  (-1.5, 1.5),   "%0.2f"),
        'psp_rise_time':                      ('PSP Rise Time',                     'ms',  1e3,  0.5,   [db.Synapse.psp_rise_time],                        'viridis_r',   True,   (1, 10),       "%0.2f"),
        'psp_decay_tau':                      ('PSP Decay Tau',                     'ms',  1e3,  0.01,  [db.Synapse.psp_decay_tau],                        'viridis_r',   True,   (1, 50),       "%0.1f"),
        'psc_amplitude':                      ('PSC Amplitude',                     'pA',  1e12, 0.3,   [db.Synapse.psc_amplitude],                        'bwr',         False,  (-20, 20),     "%0.2g"),
        'psc_rise_time':                      ('PSC Rise Time',                     'ms',  1e3,  1,     [db.Synapse.psc_rise_time],                        'viridis_r',   True,   (.1, 6),       "%0.2f"),
        'psc_decay_tau':                      ('PSC Decay Tau',                     'ms',  1e3,  1,     [db.Synapse.psc_decay_tau],                        'viridis_r',   True,   (2, 20),       "%0.1f"),
        'latency':                            ('Latency',                           'ms',  1e3,  1,     [db.Synapse.latency],                              'viridis_r',   False,  (0.5, 2),      "%0.2f"),
        'pulse_amp_90th_percentile':          ('90th Percentile PSP Amplitude',     'mV',  1e3,  1.5,   [db.Dynamics.pulse_amp_90th_percentile],           'bwr',         False,  (-1.5, 1.5),   "%0.2f"),
        'junctional_conductance':             ('Junctional Conductance',            'nS',  1e9,  1,     [db.GapJunction.junctional_conductance],           'viridis',     False,  (0, 10),       "%0.2f"),
        'coupling_coeff_pulse':               ('Coupling Coefficient',              '',    1,    1,     [db.GapJunction.coupling_coeff_pulse],             'viridis',     False,  (0, 1),        "%0.2f"),
        'stp_initial_50hz':                   ('Paired pulse STP',                  '',    1,    1,     [db.Dynamics.stp_initial_50hz],                    'bwr',         False,  (-0.5, 0.5),   "%0.2f"),
        'stp_induction_50hz':                 ('← Facilitating  Depressing →',      '',    1,    1,     [db.Dynamics.stp_induction_50hz],                  'bwr',         False,  (-0.5, 0.5),   "%0.2f"),
        'stp_recovery_250ms':                 ('← Over-recovered  Not recovered →', '',    1,    1,     [db.Dynamics.stp_recovery_250ms],                  'bwr',         False,  (-0.2, 0.2),   "%0.2f"),
        'stp_recovery_single_250ms':          ('← Over-recovered  Not recovered →', '',    1,    1,     [db.Dynamics.stp_recovery_single_250ms],           'bwr',         False,  (-0.2, 0.2),   "%0.2f"),
        'pulse_amp_first_50hz':               ('1st PSP Amplitude @ 50Hz',          '',    1e3,  1,     [db.Dynamics.pulse_amp_first_50hz],                'bwr',         False,  (-1.5, 1.5),   "%0.2f"),
        'pulse_amp_stp_initial_50hz':         ('2nd PSP Amplitude @ 50Hz',          '',    1e3,  1,     [db.Dynamics.pulse_amp_stp_initial_50hz],          'bwr',         False,  (-1.5, 1.5),   "%0.2f"),
        'pulse_amp_stp_induction_50hz':       ('PSP Amplitude STP induced @ 50Hz',  '',    1e3,  1,     [db.Dynamics.pulse_amp_stp_induction_50hz],        'bwr',         False,  (-1.5, 1.5),   "%0.2f"),
        'pulse_amp_stp_recovery_single_250ms':('PSP Amplitude STP recovered @ 250ms','',   1e3,  1,     [db.Dynamics.pulse_amp_stp_recovery_single_250ms], 'bwr',         False,  (-1.5, 1.5),   "%0.2f"),
        'paired_event_correlation_1_2_r':     ('Paired event correlation 1:2',      '',    1,    1,     [db.Dynamics.paired_event_correlation_1_2_r],      'bwr',         False,  (-0.2, 0.2),   "%0.2f"),
        'paired_event_correlation_2_4_r':     ('Paired event correlation 2:4',      '',    1,    1,     [db.Dynamics.paired_event_correlation_2_4_r],      'bwr',         False,  (-0.2, 0.2),   "%0.2f"),
        'paired_event_correlation_4_8_r':     ('Paired event correlation 4:8',      '',    1,    1,     [db.Dynamics.paired_event_correlation_4_8_r],      'bwr',         False,  (-0.2, 0.2),   "%0.2f"),
        'junctional_conductance':             ('Junctional Conductance',            'nS',  1e9,  1,     [db.GapJunction.junctional_conductance],           'viridis',     False,  (0, 10),       "%0.2f"),
        'coupling_coeff_pulse':               ('Coupling Coefficient',              '',    1,    1,     [db.GapJunction.coupling_coeff_pulse],             'viridis',     False,  (0, 1),        "%0.2f"),
        'variability_resting_state':          ('log(Resting state aCV)',       '',    1,    1,     [db.Dynamics.variability_resting_state],           'viridis',     False,  (-1, 1),       "%0.2f"),
        'variability_second_pulse_50hz':      ('log(second pulse aCV)',         '',    1,    1,    [db.Dynamics.variability_second_pulse_50hz],       'viridis',     False,  (-1, 1),       "%0.2f"),
        'variability_stp_induced_state_50hz': ('log(STP induced aCV)',         '',    1,    1,     [db.Dynamics.variability_stp_induced_state_50hz],  'viridis',     False,  (-1, 1),       "%0.2f"),
    } 
    if metrics is None:
        metrics = synapse_metrics
    
    pair_query_args = pair_query_args or {}

    metric_name, units, scale, alpha, columns, cmap, cmap_log, clim, cell_fmt = metrics[metric]

    if pre_classes is None or post_classes is None:
        return None, metric_name, units, scale, alpha, cmap, cmap_log, clim, cell_fmt

    pairs = db.matrix_pair_query(
        pre_classes=pre_classes,
        post_classes=post_classes,
        pair_query_args=pair_query_args,
        columns=columns,
    )

    pairs_has_metric = pairs[~pairs[metric].isnull()]
    pairs_has_metric = pairs_has_metric.drop_duplicates()
    return pairs_has_metric, metric_name, units, scale, alpha, cmap, cmap_log, clim, cell_fmt

def pair_class_metric_scatter(metrics, db, pair_classes, pair_query_args, ax, palette='muted', estimator=np.mean, plot_args={}):
    """To create scatter plots from get_metric_data for specific pair_classes. In this case pair_classes is a list of
    tuples of specific pre->post class pairs instead of all combos from a list of pre-classes
    and post-classes

    Parameters
    -----------
    metrics : list 
        correspond to keys in metrics dict of `get_metric_data`
    db : SynPhys database 
    pair_classes : list
        list of tuples of CellClass (pre_class, post_class)
    pair_query_args : dict
        arguments to pass to db.pair_query
    ax : matplotlib.axes
        The matplotlib axes object on which to draw the swarm plots
    palette : seaborn color palette to use for plots or list of colors that has the same length as pair_classes
    **plot_args: other arguments passed to matplotlib.plt.plot

    Outputs
    --------
    Vertically stacked scatter plots for each metric (y) and pair_class (x)
    """
    pre_classes = {pair_class[0].name: pair_class[0] for pair_class in pair_classes}
    post_classes = {pair_class[1].name: pair_class[1] for pair_class in pair_classes}
    pair_classes = ['%s→%s' % (pc[0], pc[1]) for pc in pair_classes]
    for i, metric in enumerate(metrics):
        pairs_has_metric, metric_name, units, scale, alpha, cmap, cmap_log, clim, cell_fmt = get_metric_data(metric, db, pre_classes=pre_classes, post_classes=post_classes, pair_query_args=pair_query_args)
        pairs_has_metric['pair_class'] = pairs_has_metric['pre_class'] + '→' + pairs_has_metric['post_class']
        pairs_has_metric = pairs_has_metric[pairs_has_metric['pair_class'].isin(pair_classes)]
        pairs_has_metric[metric] *= scale
        groups = pairs_has_metric.groupby('pair_class')
        y_vals = [groups.get_group(pair_class)[metric].to_list() for pair_class in pair_classes if pair_class in groups]
        if isinstance(palette, str):
            colors = sns.color_palette(palette, n_colors=len(pair_classes))
        else:
            colors = palette
        c2 = [[c]*len(y_vals[i]) for i, c in enumerate(colors)]
        if cmap_log:
            x_vals = swarm([[np.log(y) for y in group] for group in y_vals])
            ax[i].set_yscale('log')
        else:
            x_vals = swarm(y_vals)
            
        plot = sns.barplot(x='pair_class', y=metric, data=pairs_has_metric, ax=ax[i], ci=None, 
                           facecolor=(1, 1, 1, 0), edgecolor='black', order=pair_classes, estimator=estimator)
        ax[i].scatter(np.concatenate(x_vals), np.concatenate(y_vals), color=np.concatenate(c2), **plot_args)
        
        if i == len(metrics) - 1:
            ax[i].set_xlabel('pre→post class', size=12)
            ax[i].set_xticklabels(plot.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize='medium')
        else:
            ax[i].set_xlabel('')
            ax[i].set_xticklabels('')
        label = metric_name + (' (%s)'%units if units else '')
        label = '\n'.join(wrap(label, 20))
        ax[i].set_ylabel(label, size=10)
        ax[i].set_yticklabels([], minor=True)
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['top'].set_visible(False)
        ax[i].yaxis.set_ticks_position('left')
        if 'Amp' in metric_name:
            ax[i].axhline(y=0, color='k', linewidth=1)
            ax[i].spines['bottom'].set_visible(False)
            ax[i].tick_params(axis='x', bottom=False, top=False)
        else:
            ax[i].xaxis.set_ticks_position('bottom')


def metric_stats(metric, db, pre_classes, post_classes, pair_query_args):
    pairs_has_metric, _, units, scale, _, _, _, _, _ = get_metric_data(metric, db, pre_classes=pre_classes, post_classes=post_classes, pair_query_args=pair_query_args)
    pairs_has_metric[metric] = pairs_has_metric[metric].apply(pd.to_numeric)*scale
    summary = pairs_has_metric.groupby(['pre_class', 'post_class']).describe(percentiles=[0.25, 0.5, 0.75])
    return summary[metric], units


def ei_hist_plot(ax, metric, bin_edges, db, pair_query_args):
    ei_classes = {'ex': CellClass(cell_class='ex'), 'in': CellClass(cell_class='in')}
    
    pairs_has_metric, metric_name, units, scale, _, _, log_scale, _, _ = get_metric_data(metric, db, ei_classes, ei_classes, pair_query_args=pair_query_args)
    ex_pairs = pairs_has_metric[pairs_has_metric['pre_class']=='ex']
    in_pairs = pairs_has_metric[pairs_has_metric['pre_class']=='in']
    if 'amp' in metric:
        ax[0].hist(ex_pairs[metric]*scale, bins=bin_edges, color=(0.8, 0.8, 0.8), label='All Excitatory Synapses')
        ax[1].hist(in_pairs[metric]*scale, bins=bin_edges, color=(0.8, 0.8, 0.8), label='All Inhibitory Synapses')
    else:
        ax[0].hist(pairs_has_metric[metric]*scale, bins=bin_edges, color=(0.8, 0.8, 0.8), label='All Synapses')
        ax[1].hist(pairs_has_metric[metric]*scale, bins=bin_edges, color=(0.8, 0.8, 0.8), label='All Synapses')

    ee_pairs = ex_pairs[ex_pairs['post_class']=='ex']
    ei_pairs = ex_pairs[ex_pairs['post_class']=='in']
    ax[0].hist(ee_pairs[metric]*scale, bins=bin_edges, color='red', alpha=0.6, label='E->E Synapses')
    ax[0].hist(ei_pairs[metric]*scale, bins=bin_edges, color='pink', alpha=0.8, label='E->I Synapses')
    ax[0].legend(frameon=False)

    ii_pairs = in_pairs[in_pairs['post_class']=='in']
    ie_pairs = in_pairs[in_pairs['post_class']=='ex']
    ax[1].hist(ii_pairs[metric]*scale, bins=bin_edges, color='blue', alpha=0.4, label='I->I Synapses')
    ax[1].hist(ie_pairs[metric]*scale, bins=bin_edges, color='purple', alpha=0.4, label='I->E Synapses')
    ax[1].legend(frameon=False)
    
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[1].set_xlabel(metric_name + (' (%s)'%units if units else ''))
    ax[1].set_ylabel('Number of Synapses', fontsize=12)

    #KS test
    excitatory = stats.ks_2samp(ee_pairs[metric], ei_pairs[metric])
    inhibitory = stats.ks_2samp(ii_pairs[metric], ie_pairs[metric])
    print('Two-sample KS test for %s' % metric)
    print('Excitatory: p = %0.3e' % excitatory[1])
    print('Inhibitory: p = %0.3e' % inhibitory[1])

    return ex_pairs, in_pairs

def cell_class_matrix(pre_classes, post_classes, metric, class_labels, ax, db, pair_query_args=None, estimator=np.mean, clim=None):
    if class_labels is None:
        class_labels = {key: key for key in pre_classes.keys()}
    pairs_has_metric, metric_name, units, scale, alpha, cmap, cmap_log, default_clim, cell_fmt = get_metric_data(metric, db, pre_classes, post_classes, pair_query_args=pair_query_args)
    metric_data = pairs_has_metric.groupby(['pre_class', 'post_class']).aggregate(lambda x: estimator(x))
    error = pairs_has_metric.groupby(['pre_class', 'post_class']).aggregate(lambda x: np.std(x))
    count = pairs_has_metric.groupby(['pre_class', 'post_class']).count()

    shape = (len(pre_classes), len(post_classes))
    data = np.zeros(shape)
    data_alpha = np.zeros(shape)
    data_str = np.zeros(shape, dtype=object)

    for i, pre_class in enumerate(pre_classes):
        for j, post_class in enumerate(post_classes):
            try:
                value = getattr(metric_data.loc[pre_class].loc[post_class], metric)
                n = getattr(count.loc[pre_class].loc[post_class], metric)
                std = getattr(error.loc[pre_class].loc[post_class], metric)
                if n == 1:
                    value = np.nan
            except KeyError:
                value = np.nan
            data[i, j] = value * scale
            data_str[i, j] = cell_fmt % (value * scale) if np.isfinite(value) else ""
            data_alpha[i, j] = 1-alpha*((std*scale)/np.sqrt(n)) if np.isfinite(value) else 0 

    pre_labels = [class_labels[cls] for cls in pre_classes]
    post_labels = [class_labels[cls] for cls in post_classes]

    clim = clim or default_clim
    cmap = matplotlib.cm.get_cmap(cmap)
    if cmap_log:
        norm = matplotlib.colors.LogNorm(vmin=clim[0], vmax=clim[1], clip=False)
    else:
        norm = matplotlib.colors.Normalize(vmin=clim[0], vmax=clim[1], clip=False)

    mapper = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    data_rgb = mapper.to_rgba(data)
    data_rgb[:,:,3] = np.clip(data_alpha, 0, 1)

    im, cbar = heatmap(
        data_rgb, pre_labels, post_labels,
        ax=ax,
        ax_labels=('postsynaptic', 'presynaptic'),
        bg_color=(0.8, 0.8, 0.8),
        cmap=cmap,
        norm=norm,
        cbarlabel=metric_name + (' (%s)'%units if units else ''),
        cbar_kw={'shrink':0.5, 'pad':0.02},
    )

    text = annotate_heatmap(im, data_str, data=data, fontsize=8)

    return pairs_has_metric


def get_pair(expt_id, pre_cell, post_cell, db):
    expt = db.query(db.Experiment).filter(db.Experiment.ext_id==expt_id).all()[0]
    pairs = expt.pairs
    pair = pairs[(pre_cell, post_cell)]
    return pair


def map_color_by_metric(pair, metric, cmap, norm, scale):
    synapse = pair.synapse
    try:
        value= getattr(synapse, metric)*scale
    except:
        dynamics =pair.dynamics
        value = getattr(dynamics, metric)*scale
    mapper = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    color = mapper.to_rgba(value)
    return color


def plot_metric_pairs(pair_list, metric, db, ax, align='pulse', norm_amp=None, perc=False, labels=None, max_ind_freq=50):
    pairs = [get_pair(eid, pre, post, db) for eid, pre, post in pair_list]
    _, metric_name, units, scale, _, cmap, cmap_log, clim, _ = get_metric_data(metric, db)
    cmap = matplotlib.cm.get_cmap(cmap)
    if cmap_log:
        norm = matplotlib.colors.LogNorm(vmin=clim[0], vmax=clim[1], clip=False)
    else:
        norm = matplotlib.colors.Normalize(vmin=clim[0], vmax=clim[1], clip=False)
    colors = [map_color_by_metric(pair, metric, cmap, norm, scale) for pair in pairs]
    for i, pair in enumerate(pairs):
        s = db.session()
        q= response_query(s, pair, max_ind_freq=max_ind_freq)
        prs = [q.PulseResponse for q in q.all()]
        sort_prs = sort_responses(prs)
        prs = sort_prs[('ic', -55)]['qc_pass']
        if pair.synapse.synapse_type=='ex':
            prs = prs + sort_prs[('ic', -70)]['qc_pass']
        if perc:
            prs_amp = [abs(pr.pulse_response_fit.fit_amp) for pr in prs if pr.pulse_response_fit is not None]
            amp_85, amp_95 = np.percentile(prs_amp, [85, 95])
            mask = (prs_amp >= amp_85) & (prs_amp <= amp_95)
            prs = np.asarray(prs)[mask]
        prl = PulseResponseList(prs)
        post_ts = prl.post_tseries(align='spike', bsub=True, bsub_win=0.1e-3)
        trace = post_ts.mean()*scale
        if norm_amp=='exc':
            
            trace = post_ts.mean()/pair.synapse.psp_amplitude
        if norm_amp=='inh':
            trace = post_ts.mean()/pair.synapse.psp_amplitude*-1
        latency = pair.synapse.latency
        if align=='pulse':
            trace.t0 = trace.t0 - latency
        label = labels[i] if labels is not None else None
        ax.plot(trace.time_values*scale, trace.data, color=colors[i], linewidth=2, label=label)
    ax.set_xlim(-2, 10)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if labels is not None:
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))


def show_distance_profiles(ax, results, colors, class_labels):
    """ Display connection probability vs distance plots
    Parameters
    -----------
    ax : matplotlib.axes
        The matplotlib axes object on which to make the plots
    results : dict
        Output of aisynphys.connectivity.measure_distance. This structure maps
        (pre_class, post_class) onto the results of the connectivity as a function of distance.
    colors: dict
        color to draw each (pre_class, post_class) connectivity profile. Keys same as results.
        To color based on overall connection probability use color_by_conn_prob.
    class_labels : dict
        Maps {cell_class: label} to give the strings to display for each cell class.
    """

    for i, (pair_class, result) in enumerate(results.items()):
        pre_class, post_class = pair_class
        plot = ax[i]
        xvals = result['bin_edges']
        xvals = (xvals[:-1] + xvals[1:])*0.5e6
        cp = result['conn_prob']
        lower = result['lower_ci']
        upper = result['upper_ci']

        color = colors[pair_class]
        color2 = list(color)
        color2[-1] = 0.2
        mid_curve = plot.plot(xvals, cp, color=color, linewidth=2.5)
        lower_curve = plot.fill_between(xvals, lower, cp, color=color2)
        upper_curve = plot.fill_between(xvals, upper, cp, color=color2)
        
        plot.set_title('%s -> %s' % (class_labels[pre_class], class_labels[post_class]))
        if i == len(ax)-1:
            plot.set_xlabel('Distance (um)')
            plot.set_ylabel('Connection Probability')
        
    return ax


def show_connectivity_profile(x_probed, conn, ax, fit=None, true_model=None, ymax=None, fit_label=None, show_labels=False, x_bins=None):
    # where to bin connections for measuring connection probability
    if x_bins is None:
        x_bins = np.arange(0, 500e-6, 40e-6)

    # where to sample models
    #x_vals = 0.5 * (x_bins[1:] + x_bins[:-1])
    x_vals = np.linspace(x_bins[0], x_bins[-1], 200)
   
    _, cprop, lower, upper = connectivity_profile(conn, x_probed, x_bins)
    # plot the connectivity profile with confidence intervals (black line / grey area)
    show_distance_binned_cp(x_bins, cprop, ax, ci_lower=lower, ci_upper=upper)

    if ymax is None:
        ymax = upper.max()

    if fit is not None:
        show_connectivity_fit(x_vals, fit, ax, true_model=true_model, label=fit_label)

    tickheight = ymax / 10
    show_connectivity_raster(x_probed, conn, tickheight, ax)

    if show_labels is True:
        err = 0 if not hasattr(fit, 'fit_result') else fit.fit_result.fun
        label = "Fit pmax=%0.2f\nsize=%0.2f µm\nerr=%f" % (fit.pmax, fit.size*1e6, err)
        ax.text(0.99, 0.85, label, transform=ax.transAxes, color=(0.5, 0, 0), horizontalalignment='right')
        
        if true_model is not None:
            label = "True pmax=%0.2f\nsize=%0.2f µm" % (true_model.pmax, true_model.size*1e6)
            ax.text(0.99, 0.95, label, transform=ax.transAxes, color=(0, 0.5, 0), horizontalalignment='right')
    
    ax.axhline(0, color=(0, 0, 0))
    set_distance_xticks(x_vals, ax)

    y_vals = np.arange(0, ymax + 0.1, 0.1)
    ax.set_yticks([-tickheight*2, -tickheight] + list(y_vals))
    ax.set_yticklabels(['probed', 'connected'] + ['%0.1f'%x for x in y_vals])
    ax.set_ylim(-tickheight*2.6, ymax)


def show_connectivity_fit(x_vals, fit, ax, color=(0.5, 0, 0), true_model=None, label=None):
    if true_model is not None:
        # plot the ground-truth probability distribution (solid green)
        ax.plot(x_vals, true_model.connection_probability(x_vals), color=(0, 0.5, 0))
    ax.plot(x_vals, fit.connection_probability(x_vals), color=color, label=label)
    if label is not None:
        ax.legend()


def show_distance_binned_cp(x_bins, cprop, ax, color=(0.5, 0.5, 0.5), ci_lower=None, ci_upper=None):
    ax.plot(x_bins, np.append(cprop, cprop[-1]), drawstyle='steps-post', color=color)
    if ci_lower is not None and ci_upper is not None:
        ax.fill_between(x_bins, np.append(ci_lower, ci_lower[-1]), np.append(ci_upper, ci_upper[-1]), step='post', facecolor=color + (0.3,))


def show_connectivity_raster(x_probed, conn, tickheight, ax, color=(0, 0, 0), offset=2):
    # plot connections probed and found
    # warning: some mpl versions have a bug that causes the data argument to eventplot to be modified
    alpha1 = np.clip(30 / len(x_probed), 1/255, 1)
    alpha2 = np.clip(30 / conn.sum(), 1/255, 1)
    ax.eventplot(x_probed.copy(), lineoffsets=-tickheight*offset, linelengths=tickheight, color=(color + (alpha1,)))
    ax.eventplot(x_probed[conn], lineoffsets=-tickheight*(offset-1), linelengths=tickheight, color=(color + (alpha2,)))


def set_distance_xticks(x_vals, ax, interval=50e-6):
    ax.set_xlabel('distance (µm)')
    xticks = np.arange(0, x_vals.max(), interval)
    ax.set_xticks(xticks)
    ax.set_xticklabels(['%0.0f'%(x*1e6) for x in xticks])


def color_by_conn_prob(pair_group_keys, connectivity, norm, cmap):
    """ Return connection probability mapped color from show_connectivity_matrix
    """
    colors = {}
    for key in pair_group_keys:
        cp = connectivity[key]['connection_probability'][0]
        mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        color = mapper.to_rgba(np.clip(cp, 0.01, 1.0))
        colors[key] = color

    return colors


def data_matrix(data_df, cell_classes, metric=None, scale=1, unit=None, cmap=None, norm=None, alpha=2):
    """ Return data and labels to make a matrix using heatmap and annotate_heatmap. Similar to 
    show_connectivity_matrix but for arbitrary data metrics.

    Parameters:
    -----------
    data_df : pandas dataframe 
        pairs with various metrics as column names along with the pre-class and post-class.
    cell_classes : list 
        cell classes included in the matrix, this assumes a square matrix.
    metric : str
        data metric to be displayed in matrix
    scale : float
        scale of the data
    unit : str
        unit for labels
    cmap : matplotlib colormap instance
        used to colorize the matrix
    norm : matplotlib normalize instance
        used to normalize the data before colorizing
    alpha : int
        used to desaturate low confidence data
    """

    shape = (len(cell_classes), len(cell_classes))
    data = np.zeros(shape)
    data_alpha = np.zeros(shape)
    data_str = np.zeros(shape, dtype=object)
    
    mean = data_df.groupby(['pre_class', 'post_class']).aggregate(lambda x: np.mean(x))
    error = data_df.groupby(['pre_class', 'post_class']).aggregate(lambda x: np.std(x))
    count = data_df.groupby(['pre_class', 'post_class']).count()
    
    for i, pre_class in enumerate(cell_classes):
        for j, post_class in enumerate(cell_classes):
            try:
                value = mean.loc[pre_class].loc[post_class][metric]
                std = error.loc[pre_class].loc[post_class][metric]
                n = count.loc[pre_class].loc[post_class][metric]
                if n == 1:
                    value = np.nan
                #data_df.loc[pre_class].loc[post_class][metric]
            except KeyError:
                value = np.nan
            data[i, j] = value*scale
            data_str[i, j] = "%0.2f %s" % (value*scale, unit) if np.isfinite(value) else ""
            data_alpha[i, j] = 1-alpha*((std*scale)/np.sqrt(n)) if np.isfinite(value) else 0 

    mapper = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    data_rgb = mapper.to_rgba(data)
    max = mean[metric].max()*scale
    data_rgb[:,:,3] = np.clip(data_alpha, 0, max)
    return data_rgb, data_str


def plot_stim_sorted_pulse_amp(pair, ax, ind_f=50, avg_line=False, avg_trace=False, scatter_args={}, line_args={}):
    qc_pass_data = stim_sorted_pulse_amp(pair)

    # scatter plots of event amplitudes sorted by pulse number 
    mask = qc_pass_data['induction_frequency'] == ind_f
    filtered = qc_pass_data[mask].copy()

    sign = 1 if pair.synapse.synapse_type == 'ex' else -1
    try:
        filtered['dec_fit_reconv_amp'] *= sign * 1000
    except KeyError:
        print('No fit amps for pair: %s' % pair)
    ax.set_ylim(0, filtered['dec_fit_reconv_amp'].max())
    ax.set_xlim(0, 13)

    scatter_opts = {'color': (0.7, 0.7, 0.7, ), 'size': 3}
    scatter_opts.update(scatter_args)
    sns.swarmplot(x='pulse_number', y='dec_fit_reconv_amp', data=filtered, ax=ax, **scatter_opts)
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()
    

    line_opts = {'color': 'k', 'linewidth': 2, 'zorder': 100}
    line_opts.update(line_args)
    # plot a line at the average of all pulses of the same number
    if avg_line:
        pulse_means = filtered.groupby('pulse_number').mean()['dec_fit_reconv_amp'].to_list()
        ax.plot(range(0,8), pulse_means[:8], **line_opts)
        ax.plot(range(8,12), pulse_means[8:12], **line_opts)
    # plot avg trace for each pulse number
    if avg_trace:
        for pulse_number in np.arange(1,13):
            pulse_ids = filtered[filtered['pulse_number']==pulse_number]['id'].to_list()
            prs = db.query(db.PulseResponse).filter(db.PulseResponse.id.in_(pulse_ids))
            pr_list = PulseResponseList(prs)
            post_trace = pr_list.post_tseries(align='spike', bsub=True, bsub_win=1e-3)
            trace_mean = post_trace.mean()*1e3
            trace_slice = trace_mean.time_slice(-1e-3, 8e-3)
            ax.plot(trace_slice.time_values*1e2 + (pulse_number-1.4), abs(trace_slice.data),  **line_opts)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def swarm(groups, width=0.7, spacing=1.0, shared_scale=True):
    """Helper function for generating swarm plots.

    Given groups of y values to be show in a swarm plot, return appropriate x values.

    Parameters
    ----------
    groups : list
        List of y-value groups; each group is a list of y values that will appear together in a swarm.
    width : float
        The fraction of the total x-axis width to fill
    spacing : float
        The x-axis distance between adjacent groups
    shared_scale : bool
        If True, then the x values in all groups are scaled by the same amount. If False, then each group
        is scaled independently such that all groups attempt to fill their alloted width.
    """
    from pyqtgraph import pseudoScatter
    x_grps = []
    for i, y_grp in enumerate(groups):
        y_grp = np.asarray(y_grp)
        mask = np.isfinite(y_grp)
        x = np.empty(y_grp.shape)
        x[mask] = pseudoScatter(y_grp[mask], method='histogram', bidir=True)
        x = x * (0.5 * width * spacing / np.abs(x).max()) + spacing * i
        x[~mask] = np.nan
        x_grps.append(x)

    return x_grps


def fig_to_svg(fig):
    """Return a BytesIO that contains *fig* rendered to SVG.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='svg')
    buf.seek(0)
    return buf


def compose_svg_figure(figure_spec, filename, size, display=False):
    """Compose a figure from multiple SVG components.

    Parameters
    ----------
    figure_spec : dict
        Structure describing subfigures to compose (see below).
    filename : str
        File name to save composed SVG
    size : tuple
        Size of final SVG document like ("16cm", "10cm")
    display : bool
        If True, display the complete SVG in Jupyter notebook.

    Each item in *figure_spec* is a dict containing::

        {
            'figure': <matplotlib Figure>,
            'pos': (x, y),
            'scale': 1.0,
            'label': 'A',
            'label_opts': {'size': 16, 'weight': 'bold'},
        }
    """
    import svgutils.transform as svg

    fig = svg.SVGFigure(*size)

    for item in figure_spec:
        subfig = svg.from_mpl(item['figure'], savefig_kw={'bbox_inches':'tight', 'pad_inches':0})
        root = subfig.getroot()
        root.moveto(item['pos'][0], item['pos'][1], scale=item.get('scale', 1.0))
        label = svg.TextElement(item['pos'][0], item['pos'][1], item['label'], **item.get('label_opts', {}))
        fig.append([root, label])

    fig.save(filename)

    if display:
        from IPython.display import SVG, display
        display(SVG(filename=filename))


def make_scatter_legend(ax, values, cmap, norm, label_formatter, title, 
                        color=None, markersize=10, linewidth=0, markeredgewidth=0,
                        loc='upper left', bbox_to_anchor=(1, 1), marker='o',
                        **kwds):
    if isinstance(cmap, str):
        cmap = matplotlib.cm.get_cmap(cmap)

    legend_elements = []
    for x in values:
        legend_elements.append(matplotlib.lines.Line2D(
            [0], [0], marker=marker, color=color, linewidth=linewidth, markeredgewidth=markeredgewidth,
            label=label_formatter(x), markerfacecolor=cmap(norm(x)), markersize=markersize,
        ))

    return ax.legend(
        handles=legend_elements, 
        loc=loc, 
        title=title,
        bbox_to_anchor=bbox_to_anchor,
        **kwds
    )
