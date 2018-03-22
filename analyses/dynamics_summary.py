from synapse_comparison import load_cache, summary_plot_pulse
from rep_connections import ee_connections
from manuscript_figures import colors_mouse, feature_kw
from scipy import stats
import numpy as np
import pyqtgraph as pg
from neuroanalysis.ui.plot_grid import PlotGrid
from neuroanalysis.synaptic_release import ReleaseModel

app = pg.mkQApp()
pg.dbg()
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

cache_file = 'train_amps.pkl'
data = load_cache(cache_file)
model_amps = load_cache('model_amps.pkl')
feature_plt_ind = None
feature_plt_rec = None
symbols = ['d', 's', 'o', '+', 't']

model = ReleaseModel()
model.Dynamics = {'Dep':1, 'Fac':0, 'UR':0, 'SMR':0, 'DSR':0}
params = {(('2/3', 'unknown'), ('2/3', 'unknown')): [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          ((None,'rorb'), (None,'rorb')): [0, 506.7, 0, 0, 0, 0, 0.22, 0, 0, 0, 0, 0],
          ((None,'sim1'), (None,'sim1')): [0, 1213.8, 0, 0, 0, 0, 0.17, 0, 0, 0, 0, 0],
          ((None,'tlx3'), (None,'tlx3')): [0, 319.4, 0, 0, 0, 0, 0.16, 0, 0, 0, 0, 0]}

ind = data[0]
rec = data[1]
freq = 10
delta =250
order = ee_connections.keys()

ind_plt = PlotGrid()
ind_plt.set_shape(4, 1)
ind_plt.show()
rec_plt = PlotGrid()
rec_plt.set_shape(1, 2)
rec_plt.show()
ind_plt_scatter = pg.plot()
ind_plt_all = PlotGrid()
ind_plt_all.set_shape(1, 2)
ind_plt_all.show()
ind_50 = {}
ninth_pulse_250 = {}

for t, type in enumerate(order):
    if type == ((None, 'ntsr1'), (None, 'ntsr1')):
        continue
    pulse_ratio = {}
    median_ratio = []
    sd_ratio = []
    for f, freqs in enumerate(ind[type]):
        # spike_times = (np.arange(12)*1/float(freqs[0])) * 1e3
        # spike_times[8:] += 250-(1/float(freqs[0]))*1e3
        # model_eval = model.eval(list(spike_times), params[type])
        pulse_ratio[freqs[0]] = np.asarray([freqs[1][n, :]/freqs[1][n, 0] for n in range(freqs[1].shape[0])])
        avg_ratio = np.mean(pulse_ratio[freqs[0]], 0)
        sd_ratio = np.std(pulse_ratio[freqs[0]], 0)
        ind_plt[t, 0].addLegend()
        ind_plt[t, 0].plot(avg_ratio, pen=colors_mouse[t], symbol=symbols[f], symbolSize=10, symbolPen='k',
                        symbolBrush=colors_mouse[t], name=('  %d Hz' % freqs[0]))
        # if type != (('2/3', 'unknown'), ('2/3', 'unknown')):
        #     model = model_amps[type][0][f]
        #     ind_plt[t, 0].plot(model, pen=colors_mouse[t])
        ind_plt[t, 0].setXRange(0, 11)
        ind_plt[t, 0].setYRange(0, 1.5)
        color2 = (colors_mouse[t][0], colors_mouse[t][1], colors_mouse[t][2], 150)
        vals = np.hstack(pulse_ratio[freqs[0]][0])
        x = pg.pseudoScatter(vals, spacing=0.15)
        ind_plt_all[0, 1].plot(x, vals, pen=None, symbol=symbols[f], symbolSize=8, symbolPen=colors_mouse[t], symbolBrush=color2)
        ind_plt_all[0, 1].setLabels(left=['8:1 Ratio', ''])
        ind_plt_all[0, 0].plot([f], [avg_ratio[7]], pen=None, symbol='o', symbolSize=15, symbolPen='k', symbolBrush=colors_mouse[t])
        err = pg.ErrorBarItem(x=np.asarray([f]), y=np.asarray([avg_ratio[7]]), height=np.asarray([sd_ratio[7]]), beam=0.1)
        ind_plt_all[0, 0].addItem(err)
    ind_plt_all[0, 0].getAxis('bottom').setTicks([[(0, '10'), (1, '20'), (2, '50'), (3, '100')]])
    ind_plt_all[0, 0].setLabels(left=['8:1 Ratio', ''], bottom=['Frequency', 'Hz'])

    feature_list = (pulse_ratio[10][:, 7], pulse_ratio[20][:, 7], pulse_ratio[50][:, 7], pulse_ratio[100][:, 7])
    labels = [['8:1 pulse ratio', ''], ['8:1 pulse ratio', ''], ['8:1 pulse ratio', ''], ['8:1 pulse ratio', '']]
    titles = ['10Hz', '20Hz', '50Hz', '100Hz']
    feature_plt_ind = summary_plot_pulse(feature_list, labels, titles, t, plot=feature_plt_ind,
                                         color=colors_mouse[t], name=type)
    ind_50[type] = {}
    ind_50[type][50] = pulse_ratio[50][:,7]


    ninth_pulse = {}
    ind_pulses = []
    rec_pulse_ratio = {}
    rec_avg_ratio = []
    for d, delay in enumerate(rec[type]):
        recovery = []
        rec_pulse_ratio[delay[0]] = np.asarray([delay[1][n, :]/delay[1][n, 0] for n in range(delay[1].shape[0])])
        rec_avg_ratio.append(np.mean(rec_pulse_ratio[delay[0]], 0))
        # rec_sd_ratio = np.std(rec_pulse_ratio[delay[0]], 0)
        ninth_pulse[delay[0]] = (rec_pulse_ratio[delay[0]][:, 8])
        #rec_perc.append(rec_median_amps[8]/rec_median_amps[7] * 100)
        # if type != (('2/3', 'unknown'), ('2/3', 'unknown')):
        #     model_rec = model_amps[type][1][d, 8:]
        #     rec_plt[0, 1].plot([d, d+0.2, d+0.4, d+0.6], model_rec, pen=colors_mouse[t])

    grand_rec_ratio = np.mean(np.asarray(rec_avg_ratio), 0)
    rec_plt[0, 0].plot(grand_rec_ratio[:8]/grand_rec_ratio[0], pen=colors_mouse[t], symbol='o',
                       symbolSize=10, symbolPen='k',symbolBrush=colors_mouse[t])
    # if type != (('2/3', 'unknown'), ('2/3', 'unknown')):
    #     model_rec_ind = np.median(model_amps[type][1], 0)[:8]
    #     rec_plt[0, 0].plot(model_rec_ind, pen=colors_mouse[t])
    rec_plt[0, 0].getAxis('bottom').setTicks([[(0, '0'), (1, '20'), (2, '40'), (3, '60'), (4, '80'), (5, '100'), (6, '120'),
                                               (7, '140')]])
    rec_plt[0, 0].setYRange(0, 1.5)
    ninth_pulse_avg = [np.mean(delays) for delays in ninth_pulse.values()]
    rec_plt[0, 1].plot(ninth_pulse_avg, pen=None, symbol='o', symbolSize=10, symbolPen='k',
                       symbolBrush=colors_mouse[t])
    rec_plt[0, 1].setYRange(0, 1.5)
    rec_plt[0, 1].getAxis('bottom').setTicks([[(0, '250'), (1, '500'), (2, '1000'), (3, '2000'), (4, '4000')]])

    labels = [['9:1 pulse ratio', ''], ['9:1 pulse ratio', ''], ['9:1 pulse ratio', ''], ['9:1 pulse ratio', ''], ['9:1 pulse ratio', '']]
    titles = ['250ms', '500ms', '1000ms', '2000ms', '4000ms']
    feature_list = (ninth_pulse[250], ninth_pulse[500], ninth_pulse[1000], ninth_pulse[2000], ninth_pulse[4000])
    feature_plt_rec = summary_plot_pulse(feature_list, labels, titles, t, plot=feature_plt_rec,
                                         color=colors_mouse[t], name=type)

    ninth_pulse_250[type] = {}
    ninth_pulse_250[type][250] = ninth_pulse[250]

feature_kw(50, ind_50)
feature_kw(250, ninth_pulse_250)