import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import math
from statsmodels.distributions.empirical_distribution import ECDF
from windrose import WindroseAxes
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.basemap import Basemap
from matplotlib.projections import PolarAxes
import datetime
from collections import OrderedDict
import mpl_toolkits.axisartist.grid_finder as gf
import mpl_toolkits.axisartist.floating_axes as fa
from datetime import timedelta, datetime
import matplotlib.lines as mlines
from scipy.stats import linregress, pearsonr, gaussian_kde


def plot_mod_obs_ts_diff_comparison(name_stat, date_in, date_fin, time_res, obs_file, key_obs_file, vel_mod_ts, vel_obs_ts, name_exp, timerange, time_res_xaxis, path_to_output_plot_folder):

    plotname = name_stat + '_' + date_in + '_' + date_fin + '_' + \
        time_res[0] + '_mod_obs_ts_difference_comparison.png'
    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(111)
    plt.rc('font', size=24)
    plt.title('Surface (3m) Current Velocity BIAS: ' + name_stat + ' (' +
              obs_file[key_obs_file]['lat'] + ', ' + obs_file[key_obs_file]['lon'] + ') \n Period: ' + date_in + '-' + date_fin, fontsize=29)
    
    for exp in range(len(name_exp)):
        mean_vel_bias = round(np.nanmean(
            np.array(vel_mod_ts[exp][name_stat]-np.array(vel_obs_ts[name_stat]))), 2)
        mean_vel_rmsd = round(math.sqrt(np.nanmean(
            (np.array(vel_mod_ts[exp][name_stat]-np.array(vel_obs_ts[name_stat]))**2))), 2)
        plt.plot(timerange, vel_mod_ts[exp][name_stat]-vel_obs_ts[name_stat],
                 label=name_exp[exp] + ' (BIAS: '+str(mean_vel_bias)+' m/s)', linewidth=3)
        plt.plot([], [], ' ', label=name_exp[exp] +
                 ' (RMSD: ' + str(mean_vel_rmsd)+' m/s)')

    plt.grid()
    ax.tick_params(axis='both', labelsize=26)
    
    if time_res_xaxis[1] == 'd':
        ax.xaxis.set_major_locator(
            mdates.DayLocator(interval=int(time_res_xaxis[0])))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    if time_res_xaxis[1] == 'w':
        ax.xaxis.set_major_locator(
            mdates.WeekdayLocator(interval=int(time_res_xaxis[0])))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    if time_res_xaxis[1] == 'm':
        ax.xaxis.set_major_locator(
            mdates.MonthLocator(interval=int(time_res_xaxis[0])))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
    if time_res_xaxis[1] == 'y':
        ax.xaxis.set_major_locator(
            mdates.YearLocator(interval=int(time_res_xaxis[0])))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))

    fig.autofmt_xdate()
    plt.ylabel('Velocity Difference [m/s]', fontsize=40)
    plt.xlabel('Date', fontsize=40)
    legend = plt.legend(prop={'size': 20}, framealpha=0.2)
    frame = legend.get_frame()
    frame.set_linewidth(5)  # Set the width of the legend box border
    frame.set_edgecolor('black')
    plt.savefig(path_to_output_plot_folder + plotname)
    plt.clf()


def plot_mod_obs_ts_comparison(name_stat, date_in, date_fin, time_res, obs_file, key_obs_file, vel_mod_ts, vel_obs_ts, name_exp, timerange, time_res_xaxis, path_to_output_plot_folder):

    plotname = name_stat + '_' + date_in + '_' + date_fin + \
        '_' + time_res[0] + '_mod_obs_ts_comparison.png'
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    color_list = ['blue', 'red']
    plt.rc('font', size=24)
    plt.title('Surface (3m) Current Velocity: ' + name_stat + ' (' +
              obs_file[key_obs_file]['lat'] + ', ' + obs_file[key_obs_file]['lon'] + ') \n Period: ' + date_in + '-' + date_fin, fontsize=29)
    
    for exp in range(len(name_exp)):
        mean_vel_mod = round(np.nanmean(
            np.array(vel_mod_ts[exp][name_stat])), 2)
        plt.plot(timerange, vel_mod_ts[exp][name_stat],
                 label=name_exp[exp] + ' : '+str(mean_vel_mod)+' m/s', linewidth=2, color=color_list[exp])
    mean_vel_obs = round(np.nanmean(np.array(vel_obs_ts[name_stat])), 2)
    
    plt.plot(timerange, vel_obs_ts[name_stat],
             label='Observation : '+str(mean_vel_obs)+' m/s', linewidth=2, color='green')
    plt.grid()
    ax.tick_params(axis='both', labelsize=26)
    buffer = 0.01 * (timerange[-1] - timerange[0])

# Set the x-axis limits to include the buffer
    ax.set_xlim(timerange[0] - buffer, timerange[-1] + buffer)
    if time_res_xaxis[1] == 'd':
        ax.xaxis.set_major_locator(
            mdates.DayLocator(interval=int(time_res_xaxis[0])))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    if time_res_xaxis[1] == 'w':
        ax.xaxis.set_major_locator(
            mdates.WeekdayLocator(interval=int(time_res_xaxis[0])))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    if time_res_xaxis[1] == 'm':
        ax.xaxis.set_major_locator(
            mdates.MonthLocator(interval=int(time_res_xaxis[0])))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
    if time_res_xaxis[1] == 'y':
        ax.xaxis.set_major_locator(
            mdates.YearLocator(interval=int(time_res_xaxis[0])))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))

    fig.autofmt_xdate()
    plt.ylabel('Velocity [m/s]', fontsize=40)
    plt.xlabel('Date', fontsize=40)
    legend = plt.legend(prop={'size': 30}, framealpha=0.2)
    frame = legend.get_frame()
    frame.set_linewidth(5)  # Set the width of the legend box border
    frame.set_edgecolor('black')
    plt.savefig(path_to_output_plot_folder + plotname)
    plt.clf()


def plot_mod_obs_ECDF_comparison(ii, name_stat, date_in, date_fin, time_res, obs_file, key_obs_file, vel_mod_ts, vel_obs_ts, name_exp, path_to_output_plot_folder, suffix):

    plotname = name_stat + '_' + date_in + '_' + \
        date_fin + '_' + time_res[0] + suffix
    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(111)
    plt.rc('font', size=16)
    plt.grid()
    plt.title('Surface (3m) Current Velocity ECDF: ' + name_stat + ' (' +
              obs_file[key_obs_file]['lat'] + ', ' + obs_file[key_obs_file]['lon'] + ')\n Period: ' + date_in + ' - ' + date_fin, fontsize=29)
    ax.tick_params(axis='both', labelsize=26)
    plt.xlabel('velocity [m/s]', fontsize=40)
    plt.ylabel('ECDF', fontsize=40)

    color_list = ['blue', 'red']

    ecdf_obs = ECDF(np.array(vel_obs_ts[name_stat][ii]))
    plt.axhline(y=0.5, color='black', linestyle="dashed")
    for exp in range(len(name_exp)):

        ecdf_mod = ECDF(np.array(vel_mod_ts[exp][name_stat][ii]))
        plt.plot(ecdf_mod.x, ecdf_mod.y,
                 label=name_exp[exp], linewidth=4, color=color_list[exp])
    plt.plot(ecdf_obs.x, ecdf_obs.y, label="observation",
             linewidth=4, color='green')
    plt.legend(loc='lower right', prop={'size': 40})
    plt.savefig(path_to_output_plot_folder + plotname)
    plt.clf()


def plot_windrose(direction, velocity, minimum, maximum, date_in, date_fin, name_file_substring, title_substring, output_plot_folder, ymin, ymax):
    plotname = date_in + '_' + date_fin + name_file_substring + '.png'
    fig = plt.figure()

    rect = [0.1, 0.1, 0.8, 0.8]
    hist_ax = plt.Axes(fig, rect)
    hist_ax.bar(np.array([1]), np.array([1]))
    ax = WindroseAxes.from_ax()
    turbo = plt.get_cmap('turbo')
    ax.bar(direction, velocity, normed=True, bins=np.linspace(
        minimum, maximum, 5), opening=0.8, edgecolor='white', cmap=turbo)
    # set the y-axis tick positions
    ax.set_yticks(np.linspace(ymin, ymax, 5))
    ax.set_yticklabels(['{:.2f} %'.format(x)
                       for x in np.linspace(ymin, ymax, 5)], fontsize=12)

    ax.set_title(title_substring+':' + '\n Period: ' +
                 date_in + '-' + date_fin, fontsize=12)

    legend = ax.set_legend(loc=4, bbox_to_anchor=(
        1., -0.07), prop={'size': 23}, framealpha=0.3)

    for i, label in enumerate(legend.get_texts()):
        label.set_text(label.get_text() + ' m/s')
    plt.savefig(output_plot_folder + plotname, bbox_inches=None)
    plt.close()


def plot_tot_mod_obs_ECDF_comparison(date_in, date_fin, time_res, vel_mod_ts, vel_obs_ts, name_exp, path_to_output_plot_folder, suffix):
    plotname = date_in + '_' + date_fin + '_' + time_res[0] + suffix
    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(111)
    plt.rc('font', size=16)
    plt.grid()
    plt.title('Surface (3m) Current Velocity ECDF -ALL:\n Period: ' +
              date_in + '-' + date_fin, fontsize=29)
    plt.xlabel('velocity [m/s]', fontsize=40)
    plt.ylabel('ECDF', fontsize=40)
    plt.axhline(y=0.5, color='black', linestyle="dashed")

    color_list = ['blue', 'red']

    for exp in range(len(name_exp)):
        mod_array = np.array([])
        if exp == 0:
            obs_array = np.array([])
        for name_stat in vel_mod_ts[0].keys():
            ii = np.isfinite(np.array(vel_obs_ts[name_stat]))
            mod_array = np.concatenate(
                [mod_array, np.array(vel_mod_ts[exp][name_stat][ii])])
            if exp == 0:
                obs_array = np.concatenate(
                    [obs_array, np.array(vel_obs_ts[name_stat][ii])])
        ecdf_obs = ECDF(obs_array)
        ecdf_mod = ECDF(mod_array)

        plt.plot(ecdf_mod.x, ecdf_mod.y,
                 label=name_exp[exp], linewidth=4, color=color_list[exp])

    plt.plot(ecdf_obs.x, ecdf_obs.y, label="observation",
             linewidth=4, color='green')
    ax.tick_params(axis='both', labelsize=26)
    plt.legend(loc='lower right', prop={'size': 40})
    plt.savefig(path_to_output_plot_folder + plotname)
    plt.clf()


def plot_bias_ts_comparison(date_in, date_fin, time_res, timerange, name_exp, bias_ts, time_res_xaxis, path_to_output_plot_folder):

    plotname = date_in + '_' + date_fin + '_' + \
        time_res[0] + '_bias_ts_comparison.png'
    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel('Date', fontsize=40)
    ax1.set_ylabel('BIAS [m/s]', fontsize=40)

    color_list = ['blue', 'red']
    plt.rc('font', size=8)
    plt.title('Surface (3m) Current Velocity BIAS -ALL: \n Period: ' +
              date_in + '-' + date_fin, fontsize=29)
    maximum = 0
    minimum = 0
    for exp in range(len(name_exp)):
        #maximum = max(maximum, np.nanmax(list(bias_ts[exp].values())))
        minimum = -max(maximum, np.nanmax(list(bias_ts[exp].values())))
        #minimum = min(minimum, np.nanmin(list(bias_ts[exp].values())))
        maximum = -min(minimum, np.nanmin(list(bias_ts[exp].values())))
        
        values = bias_ts[exp].values()
        negated_values = [-x for x in values]  # Negate each value in the list

        ax1.plot(timerange, negated_values, label=name_exp[exp]+' : {} m/s'.format(
            round(-np.nanmean(np.array(list(bias_ts[exp].values()))), 2)), linewidth=3, color=color_list[exp])
    ax1.axhline(y=0, color='k', linestyle='--')
    ax1.tick_params(axis='y', labelsize=26)
    winner = max(abs(maximum), abs(minimum))
    ax1.set_ylim(-winner, winner)
# Calculate a small buffer for the x-axis limits
    # You can adjust the buffer percentage as needed
    buffer = 0.01 * (timerange[-1] - timerange[0])

# Set the x-axis limits to include the buffer
    ax1.set_xlim(timerange[0] - buffer, timerange[-1] + buffer)
    if time_res_xaxis[1] == 'd':
        ax1.xaxis.set_major_locator(
            mdates.DayLocator(interval=int(time_res_xaxis[0])))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    if time_res_xaxis[1] == 'w':
        ax1.xaxis.set_major_locator(
            mdates.WeekdayLocator(interval=int(time_res_xaxis[0])))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    if time_res_xaxis[1] == 'm':
        print('ciao')
        ax1.xaxis.set_major_locator(
            mdates.MonthLocator(interval=int(time_res_xaxis[0])))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))

    if time_res_xaxis[1] == 'y':
        ax1.xaxis.set_major_locator(
            mdates.YearLocator(interval=int(time_res_xaxis[0])))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))

    fig.autofmt_xdate()
    ax1.tick_params(axis='x', labelsize=20)
    ax1.set_zorder(1)
    ax1.patch.set_visible(False)
    ax1.grid(linestyle='-')
    nticks = 8
    ax1.yaxis.set_major_locator(mpl.ticker.LinearLocator(nticks))
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%0.2f'))
    # ax1.set_yticks([0])
    ax1.grid('on')

    legend = ax1.legend(loc='upper left',  prop={'size': 20}, framealpha=0.2)
    frame = legend.get_frame()
    frame.set_linewidth(5)  # Set the width of the legend box border
    frame.set_edgecolor('black')
    plt.savefig(path_to_output_plot_folder + plotname,
                dpi=300, bbox_inches="tight")
    plt.clf()


def plot_bias_rmse_ts(date_in, date_fin, time_res, timerange, bias_ts, rmsd_ts, statistics_array, time_res_xaxis, name_exp, path_to_output_plot_folder):
    plotname = date_in + '_' + date_fin + '_' + time_res + '_bias_rmse_ts.png'
    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel('Date', fontsize=40)
    ax1.set_ylabel('BIAS [m/s]', fontsize=40, color='darkblue')
    plt.rc('font', size=8)
    plt.title('Surface (3m) Current Velocity BIAS and RMSD -ALL: \n Period: ' +
              date_in + '-' + date_fin, fontsize=29)
    lns1 = ax1.plot(timerange, list(bias_ts.values(
    )), label='BIAS: {} m/s'.format(statistics_array[0]), linewidth=3, color='darkblue')
    ax1.axhline(y=0, color='k', linestyle='--')
    ax1.tick_params(axis='y', labelsize=26, colors='darkblue')
    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('RMSD [m/s]', fontsize=40, color='orange')
    lns2 = ax2.plot(timerange, np.sqrt(list(rmsd_ts.values())), color=color,
                    label='RMSD: {} m/s'.format(statistics_array[1]), linewidth=4)
    ax2.tick_params(axis='y', labelsize=26, colors='orange')
    if time_res_xaxis[1] == 'd':
        ax1.xaxis.set_major_locator(
            mdates.DayLocator(interval=int(time_res_xaxis[0])))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    if time_res_xaxis[1] == 'w':
        ax1.xaxis.set_major_locator(
            mdates.WeekdayLocator(interval=int(time_res_xaxis[0])))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    if time_res_xaxis[1] == 'm':
        ax1.xaxis.set_major_locator(
            mdates.MonthLocator(interval=int(time_res_xaxis[0])))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
    if time_res_xaxis[1] == 'y':
        ax1.xaxis.set_major_locator(
            mdates.YearLocator(interval=int(time_res_xaxis[0])))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))

    fig.autofmt_xdate()
    ax1.set_zorder(1)
    ax1.patch.set_visible(False)
    ax1.grid(linestyle='-')
    plt.text(0.17, 0.89, name_exp[0], weight='bold',
             transform=plt.gcf().transFigure, fontsize=22)
    nticks = 8
    ax1.yaxis.set_major_locator(mpl.ticker.LinearLocator(nticks))
    ax2.yaxis.set_major_locator(mpl.ticker.LinearLocator(nticks))
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%0.2f'))
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%0.2f'))
    ax1.grid('on')
    lns = lns1+lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper left',  prop={'size': 20}, framealpha=0.2)
    plt.savefig(path_to_output_plot_folder + plotname,
                dpi=300, bbox_inches="tight")
    plt.clf()


def plot_rmse_ts_comparison(date_in, date_fin, time_res, timerange, name_exp, rmsd_ts, time_res_xaxis, path_to_output_plot_folder):
    plotname = date_in + '_' + date_fin + '_' + \
        time_res[0] + '_rmse_ts_comparison.png'
    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel('Date', fontsize=40)
    ax1.set_ylabel('RMSD [m/s]', fontsize=40)
    color_list = ['blue', 'red']
    plt.rc('font', size=8)
    plt.title('Surface (3m) Current Velocity RMSD -ALL: \n Period: ' +
              date_in + '-' + date_fin, fontsize=29)
    buffer = 0.01 * (timerange[-1] - timerange[0])

# Set the x-axis limits to include the buffer
    ax1.set_xlim(timerange[0] - buffer, timerange[-1] + buffer)

    for exp in range(len(name_exp)):
        ax1.plot(timerange, np.sqrt(list(rmsd_ts[exp].values())), label=name_exp[exp] + ' : {} m/s'.format(
            round(math.sqrt(np.nanmean(np.array(list(rmsd_ts[exp].values())))), 2)), linewidth=3, color=color_list[exp])
    ax1.tick_params(axis='y', labelsize=26)
    if time_res_xaxis[1] == 'd':
        ax1.xaxis.set_major_locator(
            mdates.DayLocator(interval=int(time_res_xaxis[0])))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    if time_res_xaxis[1] == 'w':
        ax1.xaxis.set_major_locator(
            mdates.WeekdayLocator(interval=int(time_res_xaxis[0])))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    if time_res_xaxis[1] == 'm':
        ax1.xaxis.set_major_locator(
            mdates.MonthLocator(interval=int(time_res_xaxis[0])))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
    if time_res_xaxis[1] == 'y':
        ax1.xaxis.set_major_locator(
            mdates.YearLocator(interval=int(time_res_xaxis[0])))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))

    fig.autofmt_xdate()
    ax1.tick_params(axis='x', labelsize=20)
    ax1.set_zorder(1)
    ax1.patch.set_visible(False)
    ax1.grid(linestyle='-')
    nticks = 8
    ax1.yaxis.set_major_locator(mpl.ticker.LinearLocator(nticks))
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%0.2f'))
    ax1.grid('on')

    legend = ax1.legend(loc='upper left',  prop={'size': 20}, framealpha=0.2)
    frame = legend.get_frame()
    frame.set_linewidth(5)  # Set the width of the legend box border
    frame.set_edgecolor('black')
    plt.savefig(path_to_output_plot_folder + plotname,
                dpi=300, bbox_inches="tight")
    plt.clf()


class TaylorDiagram(object):
    def __init__(self, STD, fig=None, rect=111, label='_'):
        self.STD = STD
        tr = PolarAxes.PolarTransform()
        # Correlation labels
        rlocs = np.concatenate(((np.arange(11.0) / 10.0), [0.95, 0.99]))
        tlocs = np.arccos(rlocs)  # Conversion to polar angles
        gl1 = gf.FixedLocator(tlocs)  # Positions
        tf1 = gf.DictFormatter(dict(zip(tlocs, map(str, rlocs))))
        # Standard deviation axis extent
        self.smin = 0
        self.smax = 1.6 * self.STD[0]
        gh = fa.GridHelperCurveLinear(tr, extremes=(
            0, (np.pi/2), self.smin, self.smax), grid_locator1=gl1, tick_formatter1=tf1,)
        if fig is None:
            fig = plt.figure()
        ax = fa.FloatingSubplot(fig, rect, grid_helper=gh)
        fig.add_subplot(ax)
        # Angle axis
        ax.axis['top'].set_axis_direction('bottom')
        ax.axis['top'].label.set_text("Correlation coefficient")
        ax.axis['top'].label.set_size(32)
        ax.axis['top'].major_ticklabels.set_fontsize(16)
        ax.axis['top'].toggle(ticklabels=True, label=True)
        ax.axis['top'].major_ticklabels.set_axis_direction('top')
        ax.axis['top'].label.set_axis_direction('top')
        # X axis
        ax.axis['left'].set_axis_direction('bottom')
        ax.axis['left'].label.set_text("Standard deviation")
        ax.axis['left'].label.set_size(32)
        ax.axis['left'].major_ticklabels.set_fontsize(16)
        ax.axis['left'].toggle(ticklabels=True, label=True)
        ax.axis['left'].major_ticklabels.set_axis_direction('bottom')
        ax.axis['left'].label.set_axis_direction('bottom')
        # Y axis
        ax.axis['right'].set_axis_direction('top')
        ax.axis['right'].label.set_text("Standard deviation")
        ax.axis['right'].major_ticklabels.set_fontsize(16)
        ax.axis['right'].label.set_size(32)
        ax.axis['right'].toggle(ticklabels=True, label=True)
        ax.axis['right'].major_ticklabels.set_axis_direction('left')
        ax.axis['right'].label.set_axis_direction('top')
        # Useless
        ax.axis['bottom'].set_visible(False)
        # Contours along standard deviations
        ax.grid()
        self._ax = ax  # Graphical axes
        self.ax = ax.get_aux_axes(tr)  # Polar coordinates
        # Add reference point and STD contour
        l, = self.ax.plot([0], self.STD[0], 'r*', ls='', ms=12, label=label[0])
        l1, = self.ax.plot([0], self.STD[0], 'r*',
                           ls='', ms=12, label=label[0])
#    q , = self.ax.plot([0], self.STD[1], 'b*', ls='', ms=12, label=label[1])
#    q1 , = self.ax.plot([0], self.STD[1], 'b*', ls='', ms=12, label=label[1])
        t = np.linspace(0, (np.pi / 2.0))
        t1 = np.linspace(0, (np.pi / 2.0))
        r = np.zeros_like(t) + self.STD[0]
        r1 = np.zeros_like(t) + self.STD[0]
#    p = np.zeros_like(t) + self.STD[1]
#    p1 = np.zeros_like(t) + self.STD[1]
        self.ax.plot(t, r, 'k--', label='_')
#    self.ax.plot(t, p, 'b--', label='_')
        # Collect sample points for latter use (e.g. legend)
        self.samplePoints = [l]
#    self.samplePoints = [l1]
        # self.samplePoints.append(q)
#    self.samplePoints.append(q1)

    def add_sample(self, STD, r, *args, **kwargs):
        l, = self.ax.plot(np.arccos(r), STD, *args, **
                          kwargs)  # (theta, radius)
        self.samplePoints.append(l)
        return l

#  def add_sample(self,STD,r1,*args,**kwargs):
#    l1,= self.ax.plot(np.arccos(r1), STD, *args, **kwargs) # (theta, radius)
#    self.samplePoints.append(l1)
#    return l1

    def add_contours(self, component, color, levels=5, **kwargs):
        rs, ts = np.meshgrid(np.linspace(
            self.smin, self.smax), np.linspace(0, (np.pi / 2.0)))
        RMSE = np.sqrt(np.power(self.STD[component], 2) + np.power(
            rs, 2) - (2.0 * self.STD[component] * rs * np.cos(ts)))
        contours = self.ax.contour(ts, rs, RMSE, levels, colors=color)
        return contours


def srl(obsSTD, s, r, l, fname, markers, output_plot_folder_comparison):
    fig = plt.figure(figsize=(20, 16))
    dia = TaylorDiagram(obsSTD, fig=fig, rect=111, label=['ref'])
    plt.clabel(dia.add_contours(0, 'k'), inline=1, fontsize=40)
    #plt.clabel(dia.add_contours(1,'b'), inline=1, fontsize=20)
    srlc = zip(s, r, l)
    #srlc1 = zip(s1, r1, l1)

    for count, i in enumerate(srlc):
        dia.add_sample(i[0], i[1], label=i[2], marker=markers[count],
                       markersize=12, mec='red', mfc='none', mew=1.6)
#  for count,i in enumerate(srlc1):
#    dia.add_sample(i[0], i[1], label=i[2], marker=markers[count],markersize=12, mec = 'blue', mfc = 'none', mew=1.6)
        spl = [p.get_label() for p in dia.samplePoints]
        fig.legend(dia.samplePoints, spl, numpoints=1,
                   prop={'size': 20},  loc=[0.83, 0.55])
        fig.suptitle("Taylor Diagram for " +
                     fname.split('_buoy')[0], fontsize=35)
        fig.savefig(output_plot_folder_comparison+'/'+fname)


def append_value(dict_obj, key, value):
    # Check if key exist in dict or not
    if key in dict_obj:
        # Key exist in dict.
        # Check if type of value of key is list or not
        if not isinstance(dict_obj[key], list):
            # If type is not list then make it list
            dict_obj[key] = [dict_obj[key]]
        # Append the value in list
        dict_obj[key].append(value)
    else:
        # As key is not in dict,
        # so, add key-value pair
        dict_obj[key] = value


def Get_String_Time_Resolution(start_date, end_date, time_res_to_average):
    dates = [start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")]
    start, end = [datetime.strptime(_, "%Y-%m-%d") for _ in dates]
    if time_res_to_average[-1] == 'D':
        string_time_res = list(OrderedDict(((start + timedelta(_)).strftime(
            r"%d-%b-%y"), None) for _ in range((end - start).days+1)).keys())
    if time_res_to_average[-1] == 'M':
        string_time_res = list(OrderedDict(
            ((start + timedelta(_)).strftime(r"%b-%y"), None) for _ in range((end - start).days+1)).keys())

    return string_time_res


def scatterPlot(mod, obs, outname, name, n_stations, n_time, possible_markers, hfr_name, pos_colors, time_string, **kwargs):

    if np.isnan(obs).any() or np.isnan(mod).any():

        obs_no_nan = obs[~np.isnan(obs) & ~np.isnan(mod)]
        mod_no_nan = mod[~np.isnan(obs) & ~np.isnan(mod)]
        xy = np.vstack([obs_no_nan, mod_no_nan])
    else:
        xy = np.vstack([obs, mod])

    color_list = pos_colors
    # possible_markers=np.array(["o","^","s","P","*","D"])
    if n_stations == 1:
        print("prima repeat: ", possible_markers)
        m = np.repeat(possible_markers, len(
            obs[~np.isnan(obs) & ~np.isnan(mod)]))
        c_prova = np.tile(np.arange(0, 6*len(obs), 6), 1)

    if n_stations > 1:
        m = np.array([])
        c_prova = np.tile(np.arange(0, 6*n_time, 6), n_stations)
        for stat_counter, not_nan_num in enumerate(kwargs['len_not_nan_values']):
            m_element = np.repeat(possible_markers[stat_counter], not_nan_num)
            m = np.concatenate([m, m_element])

#    c_prova = np.tile(np.arange(0,6*len(obs),6),1)
#    z = gaussian_kde(xy)(xy)
#    idx = z.argsort()

    if np.isnan(obs).any() or np.isnan(mod).any():
        #        x, y, z = obs_no_nan[idx], mod_no_nan[idx], z[idx]
        x, y = obs_no_nan, mod_no_nan
    else:
        #x, y, z = obs[idx], mod[idx], z[idx]
        x, y = obs, mod

    color_list_seq = np.tile(color_list[:n_time], n_stations)
    classes = time_string
    markers_labels = hfr_name
    fig, ax = plt.subplots(figsize=(10, 6))
#    im = ax.scatter(x, y, c=z, s=8, edgecolor=None, cmap='jet', clip_on=False)
    #im = mscatter(x, y, cmap_prova(c_prova), ax=ax, m=m, c=c_prova,s=15)
    im = mscatter(x, y, ax=ax, m=m,
                  c=c_prova[~np.isnan(obs) & ~np.isnan(mod)], s=15)
    marker_array = []
    if n_stations == 1:
        marker_array.append(mlines.Line2D([], [], color='blue', marker=possible_markers[0],
                            linestyle='None', markersize=5, label=markers_labels))
    else:
        for mark, mark_label in zip(possible_markers, markers_labels):
            marker_array.append(mlines.Line2D(
                [], [], color='blue', marker=mark, linestyle='None', markersize=5, label=mark_label))

    if n_time < 30:
        legend_1 = plt.legend(handles=im.legend_elements(num=n_time)[
                              0], labels=classes, loc='right', prop={"size": 9}, bbox_to_anchor=(1.3, 0.5))

    plt.legend(handles=marker_array, loc='upper left', prop={"size": 12})
    if n_time < 30:
        plt.gca().add_artist(legend_1)

    maxVal = np.nanmax((x, y))
#    ax.set_ylim(0, maxVal)
    ax.set_ylim(0, maxVal)
#    ax.set_xlim(0, maxVal)
    ax.set_xlim(0, maxVal)
    ax.set_aspect(1.0)
    ax.tick_params(axis='both', labelsize=12.5)

    bias = BIAS(y, x)
    print(x.shape)
    corr, _ = pearsonr(x, y)
    rmse = RMSE(y, x)
    nstd = Normalized_std(y, x)
    si = ScatterIndex(y, x)
    slope, intercept, rvalue, pvalue, stderr = linregress(y, x)

    prova = x[:, np.newaxis]
    a, _, _, _ = np.linalg.lstsq(prova, y)
    xseq = np.linspace(0, maxVal, num=100)
    ax.plot(xseq, a*xseq, 'r-')
    plt.text(0.001, 0.7, name, weight='bold',
             transform=plt.gcf().transFigure, fontsize=16)

    plt.text(0.01, 0.32, 'Entries: %s\n'
             'BIAS: %s m/s\n'
             'RMSD: %s m/s\n'
             'NSTD: %s\n'
             'SI: %s\n'
             'corr:%s\n'
             'Slope: %s\n'
             'STDerr: %s m/s'
             % (len(obs), bias, rmse, nstd, si, np.round(corr, 2),
                np.round(a[0], 2), np.round(stderr, 2)), transform=plt.gcf().transFigure, fontsize=15)

    stat_array = [bias, rmse, si, np.round(
        corr, 2), np.round(stderr, 2), len(obs)]

    if 'title' in kwargs:
        plt.title(kwargs['title'], fontsize=15, x=0.5, y=1.01)

    if 'xlabel' in kwargs:
        plt.xlabel(kwargs['xlabel'], fontsize=18)

    if 'ylabel' in kwargs:
        plt.ylabel(kwargs['ylabel'], fontsize=18)

    ax.plot([0, maxVal], [0, maxVal], c='k', linestyle='-.')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    plt.savefig(outname)
    plt.close()
    return stat_array


def QQPlot(mod, obs, outname, name, **kwargs):

    if np.isnan(obs).any() or np.isnan(mod).any():

        obs_no_nan = obs[~np.isnan(obs) & ~np.isnan(mod)]
        mod_no_nan = mod[~np.isnan(obs) & ~np.isnan(mod)]
        xy = np.vstack([obs_no_nan, mod_no_nan])
    else:
        xy = np.vstack([obs, mod])

    if np.isnan(obs).any() or np.isnan(mod).any():
        x, y = obs_no_nan, mod_no_nan
    else:
        x, y = obs, mod

    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.scatter(x, y, c=z, s=3, edgecolor=[], cmap='jet')

    maxVal = np.nanmax((x, y))
    ax.set_ylim(0, maxVal)
    ax.set_xlim(0, maxVal)
    ax.set_aspect(1.0)
    ax.tick_params(axis='both', labelsize=12.5)

    bias = BIAS(y, x)
    corr, _ = pearsonr(x, y)
    rmse = RMSE(y, x)
    nstd = Normalized_std(y, x)
    si = ScatterIndex(y, x)
    slope, intercept, rvalue, pvalue, stderr = linregress(y, x)

    prova = x[:, np.newaxis]
    a, _, _, _ = np.linalg.lstsq(prova, y)
    xseq = np.linspace(0, maxVal, num=100)
    ax.plot(xseq, a*xseq, 'r-')
    plt.text(0.001, 0.7, name, weight='bold',
             transform=plt.gcf().transFigure, fontsize=13)

    plt.text(0.01, 0.32, 'Entries: %s\n'
             'BIAS: %s m/s\n'
             'RMSD: %s m/s\n'
             'NSTD: %s\n'
             'SI: %s\n'
             'corr:%s\n'
             'Slope: %s\n'
             'STDerr: %s m/s'
             % (len(obs), bias, rmse, nstd, si, np.round(corr, 2),
                np.round(a[0], 2), np.round(stderr, 2)), transform=plt.gcf().transFigure, fontsize=15)

    stat_array = [bias, rmse, si, np.round(
        corr, 2), np.round(stderr, 2), len(obs)]

    if 'title' in kwargs:
        plt.title(kwargs['title'], fontsize=15, x=0.5, y=1.01)

    if 'xlabel' in kwargs:
        plt.xlabel(kwargs['xlabel'], fontsize=18)

    if 'ylabel' in kwargs:
        plt.ylabel(kwargs['ylabel'], fontsize=18)

    ax.plot([0, maxVal], [0, maxVal], c='k', linestyle='-.')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ticks_1 = np.linspace(z.min(), z.max(), 5, endpoint=True)
    cbar = plt.colorbar(im, fraction=0.02, ticks=ticks_1)
    cbar.ax.set_yticklabels(['{:.1f}'.format(x) for x in ticks_1], fontsize=13)
    cbar.set_label('probaility density [%]',
                   rotation=270, size=18, labelpad=15)

    plt.savefig(outname)
    plt.close()
    return stat_array


def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)+1):
        yield start_date + timedelta(n)


def mapping(obs_file, lat, lon, work_dir_plot):

    for key in obs_file.copy():
        lat[key] = obs_file[key]['lat']
        lon[key] = obs_file[key]['lon']

    plt.figure(figsize=(12, 8))
    m = Basemap(projection='cyl', llcrnrlat=30.19, urcrnrlat=45.98,
                llcrnrlon=-18.125, urcrnrlon=36.3, resolution='h')
    g = np.linspace(-0.0000001, 0.0000001, 41, endpoint=True)
    m.drawcoastlines()
    m.drawstates()
    m.drawcountries()
    m.fillcontinents(color='gray')
    # make latitude lines ever 5 degrees from 30N-50N
    parallels = np.arange(10, 50, 5.)
    # make longitude lines every 5 degrees from 95W to 70W
    meridians = np.arange(-20, 50, 5.)
    m.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=14)
    m.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=14)

    lons = [float(x) for x in list(lon.values())]
    lats = [float(x) for x in list(lat.values())]
    x, y = m(lons, lats)

    colors = plt.cm.jet(np.linspace(0, 1, len(x)))

    for count, key in enumerate(list(obs_file.keys())):
        plt.scatter(x[count], y[count], 50, alpha=1,
                    c=colors[count], label=key, zorder=2)

    plt.legend(bbox_to_anchor=(0.5, -0.87),
               loc='lower center', prop={'size': 13})
    plt.title('Location of accepted moorings', fontsize=25)
    plt.savefig(work_dir_plot+'location.png')


def line_A(x, m_A, q_A):
    return (m_A*x+q_A)


def BIAS(data, obs):
    # return np.round((np.nanmean(data-obs)).data, 2)
    return np.round((np.nanmean(obs-data)).data, 2)


def RMSE(data, obs):
    return np.round(np.sqrt(np.nanmean((data-obs)**2)), 2)


def ScatterIndex(data, obs):
    num = np.sum(((data-np.nanmean(data))-(obs-np.nanmean(obs)))**2)
    denom = np.sum(obs**2)
    return np.round(np.sqrt((num/denom)), 2)


def Normalized_std(data, obs):
    data_std = np.std(data)
    data_obs = np.std(obs)
    return np.round(data_std/data_obs, 2)


def mscatter(x, y, ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers
    ax = ax or plt.gca()
    sc = ax.scatter(x, y, clip_on=False, cmap='plasma', **kw)
    if (m is not None) and (len(m) == len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc

# UNUSED FUNCTIONS


def scatterPlot_1(mod, obs, outname, name, **kwargs):

    if np.isnan(obs).any() or np.isnan(mod).any():

        obs_no_nan = obs[~np.isnan(obs) & ~np.isnan(mod)]
        mod_no_nan = mod[~np.isnan(obs) & ~np.isnan(mod)]
        xy = np.vstack([obs_no_nan, mod_no_nan])
    else:
        xy = np.vstack([obs, mod])

    z = gaussian_kde(xy)(xy)
    idx = z.argsort()

    if np.isnan(obs).any() or np.isnan(mod).any():
        x, y, z = obs_no_nan[idx], mod_no_nan[idx], z[idx]
    else:
        x, y, z = obs[idx], mod[idx], z[idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.scatter(x, y, c=z, s=8, edgecolor=None, cmap='jet', clip_on=False)

    maxVal = np.nanmax((x, y))
    ax.set_ylim(0, maxVal)
    ax.set_xlim(0, maxVal)
    ax.set_aspect(1.0)
    ax.tick_params(axis='both', labelsize=12.5)

    bias = BIAS(y, x)
    corr, _ = pearsonr(x, y)
    rmse = RMSE(y, x)
    nstd = Normalized_std(y, x)
    si = ScatterIndex(y, x)
    slope, intercept, rvalue, pvalue, stderr = linregress(y, x)

    prova = x[:, np.newaxis]
    a, _, _, _ = np.linalg.lstsq(prova, y)
    xseq = np.linspace(0, maxVal, num=100)
    ax.plot(xseq, a*xseq, 'r-')

    plt.text(0.05, 0.7, name, weight='bold',
             transform=plt.gcf().transFigure, fontsize=18)

    plt.text(0.12, 0.32, 'Entries: %s\n'
             'BIAS: %s m/s\n'
             'RMSD: %s m/s\n'
             'NSTD: %s\n'
             'SI: %s\n'
             'corr:%s\n'
             'Slope: %s\n'
             'STDerr: %s m/s'
             % (len(obs), bias, rmse, nstd, si, np.round(corr, 2),
                np.round(a[0], 2), np.round(stderr, 2)), transform=plt.gcf().transFigure, fontsize=15)

    stat_array = [bias, rmse, si, np.round(
        corr, 2), np.round(stderr, 2), len(obs)]

    if 'title' in kwargs:
        plt.title(kwargs['title'], fontsize=20, x=0.5, y=1.01)

    if 'xlabel' in kwargs:
        plt.xlabel(kwargs['xlabel'], fontsize=18)

    if 'ylabel' in kwargs:
        plt.ylabel(kwargs['ylabel'], fontsize=18)

    ax.plot([0, maxVal], [0, maxVal], c='k', linestyle='-.')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ticks_1 = np.linspace(z.min(), z.max(), 5, endpoint=True)
    cbar = plt.colorbar(im, fraction=0.02, ticks=ticks_1)
    cbar.ax.set_yticklabels(['{:.1f}'.format(x) for x in ticks_1], fontsize=13)
    cbar.set_label('probaility density [%]',
                   rotation=270, size=18, labelpad=15)

    plt.savefig(outname)
    plt.close()
    return stat_array


def plot_mod_obs_ts_diff(name_stat, date_in, date_fin, time_res, obs_file, key_obs_file, vel_mod_ts, vel_obs_ts, name_exp, time_res_xaxis, path_to_output_plot_folder):
    plotname = name_stat + '_' + date_in + '_' + date_fin + \
        '_' + time_res + '_mod_obs_ts_difference.png'
    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(111)
    plt.rc('font', size=24)
    plt.title('Surface (3m) Current Veloity BIAS: ' + name_stat + ' (' +
              obs_file[key_obs_file]['lat'] + ', ' + obs_file[key_obs_file]['lon'] + ') \n Period: ' + date_in + '-' + date_fin, fontsize=29)
    mean_vel_bias = round(np.nanmean(
        np.array(vel_mod_ts[name_stat])-np.array(vel_obs_ts[name_stat])), 2)
    mean_vel_rmsd = round(math.sqrt(np.nanmean(
        (np.array(vel_mod_ts[name_stat])-np.array(vel_obs_ts[name_stat]))**2)), 2)

    plt.plot(timerange, vel_mod_ts[name_stat]-vel_obs_ts[name_stat],
             label='BIAS: '+str(mean_vel_bias)+' m/s', linewidth=3)
    plt.plot([], [], ' ', label='RMSD: '+str(mean_vel_rmsd)+' m/s')
    plt.grid()
    plt.text(0.17, 0.89, name_exp[0], weight='bold',
             transform=plt.gcf().transFigure, fontsize=22)
    ax.tick_params(axis='both', labelsize=26)
    if time_res_xaxis[1] == 'd':
        ax.xaxis.set_major_locator(
            mdates.DayLocator(interval=int(time_res_xaxis[0])))
    if time_res_xaxis[1] == 'w':
        ax.xaxis.set_major_locator(
            mdates.WeekdayLocator(interval=int(time_res_xaxis[0])))
    if time_res_xaxis[1] == 'm':
        ax.xaxis.set_major_locator(
            mdates.MonthLocator(interval=int(time_res_xaxis[0])))
    if time_res_xaxis[1] == 'y':
        ax.xaxis.set_major_locator(
            mdates.YearLocator(interval=int(time_res_xaxis[0])))

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    fig.autofmt_xdate()
    plt.ylabel('Velocity Difference [m/s]', fontsize=40)
    plt.xlabel('Date', fontsize=40)
    plt.legend(prop={'size': 20}, framealpha=0.2)
    plt.savefig(path_to_output_plot_folder + plotname)
    plt.clf()


def plot_mod_obs_ts(name_stat, date_in, date_fin, time_res, obs_file, key_obs_file, vel_mod_ts, vel_obs_ts, name_exp, time_res_xaxis, path_to_output_plot_folder):
    plotname = name_stat + '_' + date_in + '_' + \
        date_fin + '_' + time_res + '_mod_obs_ts.png'
    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(111)
    plt.rc('font', size=24)
    plt.title('Surface (3m) Current Velocity: ' + name_stat + ' (' +
              obs_file[key_obs_file]['lat'] + ', ' + obs_file[key_obs_file]['lon'] + ') \n Period: ' + date_in + '-' + date_fin, fontsize=29)
    mean_vel_mod = round(np.nanmean(np.array(vel_mod_ts[name_stat])), 2)
    mean_vel_obs = round(np.nanmean(np.array(vel_obs_ts[name_stat])), 2)
    tot_mean_stat = [mean_vel_mod, mean_vel_obs]
    plt.plot(timerange, vel_mod_ts[name_stat], label='Model (mean: ' +
             str(mean_vel_mod)+' m/s)', linewidth=3, color='darkblue')
    plt.plot(timerange, vel_obs_ts[name_stat], label='Observation (mean: '+str(
        mean_vel_obs)+' m/s)', linewidth=3, color='orange')
    plt.grid()
    plt.text(0.17, 0.89, name_exp[0], weight='bold',
             transform=plt.gcf().transFigure, fontsize=22)
    ax.tick_params(axis='both', labelsize=26)
    if time_res_xaxis[1] == 'd':
        ax.xaxis.set_major_locator(
            mdates.DayLocator(interval=int(time_res_xaxis[0])))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    if time_res_xaxis[1] == 'w':
        ax.xaxis.set_major_locator(
            mdates.WeekdayLocator(interval=int(time_res_xaxis[0])))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    if time_res_xaxis[1] == 'm':
        # ax.xaxis.set_major_locator(mdates.MonthLocator(interval=int(time_res_xaxis[0])))
        ax.xaxis.set_major_locator(
            mdates.MonthLocator(interval=int(time_res_xaxis[0])))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
    if time_res_xaxis[1] == 'y':
        ax.xaxis.set_major_locator(
            mdates.YearLocator(interval=int(time_res_xaxis[0])))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))

    fig.autofmt_xdate()
    plt.ylabel('Velocity [m/s]', fontsize=40)
    plt.xlabel('Date', fontsize=40)
    plt.legend(prop={'size': 30}, framealpha=0.2)
    # Save and close
    plt.savefig(path_to_output_plot_folder + plotname)
    return tot_mean_stat


def plot_depth_obs_hist(name_stat, date_in, date_fin, obs_file, key_obs_file, depth_obs_ts, path_to_output_plot_folder):
    plotname = name_stat + '_' + date_in + '_' + \
        date_fin + '_depth_obs_histogram.png'
    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(111)
    plt.rc('font', size=16)
    plt.title('Observation Depth Frequency Distribution: ' + name_stat + ' (' +
              obs_file[key_obs_file]['lat'] + ', ' + obs_file[key_obs_file]['lon'] + ')\n --- Time Period: ' + date_in + '-' + date_fin, fontsize=26)
    #min_mod = np.amin(np.array(vel_mod_ts[name_stat]))
    #max_mod = np.amax(np.array(vel_mod_ts[name_stat]))
    #min_obs = np.amin(np.array(vel_obs_ts[name_stat]))
    #max_obs = np.amax(np.array(vel_obs_ts[name_stat]))
    bins = np.linspace(np.amin(depth_obs_ts[name_stat][~np.isnan(depth_obs_ts[name_stat])]), np.amax(
        depth_obs_ts[name_stat][~np.isnan(depth_obs_ts[name_stat])]), 100)

    plt.hist(depth_obs_ts[name_stat][~np.isnan(
        depth_obs_ts[name_stat])], bins, label='depth_obs_hist')
    ax.tick_params(axis='both', labelsize=13)
    plt.xlabel('depth [m]', fontsize=40)
    plt.ylabel('frequency', fontsize=40)
    plt.legend(loc='upper right', prop={'size': 30})
    plt.savefig(path_to_output_plot_folder + plotname)


def plot_qflag_obs_hist(name_stat, date_in, date_fin, obs_file, key_obs_file, qflag_obs_ts, path_to_output_plot_folder):
    plotname = name_stat + '_' + date_in + '_' + \
        date_fin + '_qflag_obs_histogram.png'
    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(111)
    plt.rc('font', size=16)
    plt.title('Observation Quality Flag Frequency Distribution: ' + name_stat + ' (' +
              obs_file[key_obs_file]['lat'] + ', ' + obs_file[key_obs_file]['lon'] + ')\n --- Time Period: ' + date_in + '-' + date_fin, fontsize=26)
    bins = np.linspace(np.amin(qflag_obs_ts[name_stat][~np.isnan(qflag_obs_ts[name_stat])]), np.amax(
        qflag_obs_ts[name_stat][~np.isnan(qflag_obs_ts[name_stat])]), 10)
    plt.hist(qflag_obs_ts[name_stat][~np.isnan(
        qflag_obs_ts[name_stat])], bins, label='qflag_obs_hist')
    ax.tick_params(axis='both', labelsize=13)
    plt.xlabel('quality flag', fontsize=40)
    plt.ylabel('frequency', fontsize=40)
    plt.legend(loc='upper right', prop={'size': 30})
    plt.savefig(path_to_output_plot_folder + plotname)
    plt.clf()


def plot_mod_obs_hist(name_stat, date_in, date_fin, time_res, obs_file, key_obs_file, vel_mod_ts, vel_obs_ts, path_to_output_plot_folder):
    plotname = name_stat + '_' + date_in + '_' + \
        date_fin + '_' + time_res + '_mod_obs_histograms.png'
    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(111)
    plt.rc('font', size=16)
    plt.title('Surface (3m) Current Velocity Frequency Distribution: ' + name_stat + ' (' +
              obs_file[key_obs_file]['lat'] + ', ' + obs_file[key_obs_file]['lon'] + ')\n --- Time Period: ' + date_in + '-' + date_fin, fontsize=26)
    hist(vel_mod_ts[name_stat], bins="scott", ax=ax,
         histtype='stepfilled', alpha=0.5, density=True, label='model_hist')
    hist(vel_obs_ts[name_stat], bins="scott", ax=ax,
         histtype='stepfilled', alpha=0.5, density=True, label='observation')
    ax.tick_params(axis='both', labelsize=13)
    plt.xlabel('velocity [m/s]', fontsize=40)
    plt.ylabel('frequency', fontsize=40)
    plt.legend(loc='upper right', prop={'size': 30})
    plt.savefig(path_to_output_plot_folder + plotname)
    plt.clf()


def plot_mod_obs_ECDF(name_stat, date_in, date_fin, time_res, obs_file, key_obs_file, vel_mod_ts, vel_obs_ts, path_to_output_plot_folder, suffix):

    plotname = name_stat + '_' + date_in + '_' + date_fin + '_' + time_res + suffix
    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(111)
    plt.rc('font', size=16)
    plt.grid()
    plt.title('Surface (3m) Current Velocity ECDF: ' + name_stat + ' (' +
              obs_file[key_obs_file]['lat'] + ', ' + obs_file[key_obs_file]['lon'] + ')\n Period: ' + date_in + ' - ' + date_fin, fontsize=29)
    ax.tick_params(axis='both', labelsize=26)
    plt.xlabel('velocity [m/s]', fontsize=40)
    plt.ylabel('ECDF', fontsize=40)
    ecdf_obs = ECDF(np.array(vel_obs_ts))
    ecdf_mod = ECDF(np.array(vel_mod_ts))
    plt.axhline(y=0.5, color='black', linestyle="dashed")
    plt.plot(ecdf_mod.x, ecdf_mod.y, label="model", linewidth=4)
    plt.plot(ecdf_obs.x, ecdf_obs.y, label="observation", linewidth=4)
    plt.legend(loc='lower right', prop={'size': 40})
    plt.text(0.17, 0.89, name_exp[0], weight='bold',
             transform=plt.gcf().transFigure, fontsize=22)
    plt.savefig(path_to_output_plot_folder + plotname)
    plt.clf()


def plot_mod_obs_ECDF_total(date_in, date_fin, time_res, mod_array, obs_array, path_to_output_plot_folder, suffix):
    plotname = date_in + '_' + date_fin + '_' + time_res + suffix
    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(111)
    plt.rc('font', size=16)
    plt.grid()
    plt.title('Surface (3m) Current Velocity ECDF -ALL:\n Period: ' +
              date_in + '-' + date_fin, fontsize=29)
    plt.xlabel('velocity [m/s]', fontsize=40)
    plt.ylabel('ECDF', fontsize=40)
    ecdf_obs = ECDF(obs_array)
    ecdf_mod = ECDF(mod_array)
    plt.axhline(y=0.5, color='black', linestyle="dashed")
    plt.plot(ecdf_mod.x, ecdf_mod.y, label="model velocity [m/s]", linewidth=4)
    plt.plot(ecdf_obs.x, ecdf_obs.y,
             label="observation velocity [m/s]", linewidth=4)
    ax.tick_params(axis='both', labelsize=26)
    plt.legend(loc='lower right', prop={'size': 40})
    plt.text(0.17, 0.89, name_exp[0], weight='bold',
             transform=plt.gcf().transFigure, fontsize=22)
    plt.savefig(path_to_output_plot_folder + plotname)
    plt.clf()
