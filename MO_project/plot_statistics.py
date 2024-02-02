from core_plot import plot_mod_obs_ts_diff_comparison, plot_mod_obs_ts_comparison, plot_mod_obs_ECDF_comparison, plot_bias_rmse_ts, plot_windrose, plot_tot_mod_obs_ECDF_comparison, plot_bias_ts_comparison, plot_rmse_ts_comparison, append_value, srl, TaylorDiagram, append_value, Get_String_Time_Resolution, scatterPlot, daterange, mapping, QQPlot
import skill_metrics as sm
from windrose import WindroseAxes
from datetime import date
import xarray as xr
import pandas as pd
import csv
import warnings
import sys
from os.path import isfile, join
from os import listdir
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import matplotlib as mpl
mpl.use('Agg')
warnings.filterwarnings("ignore")  # Avoid warnings


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot Statistics')
    parser.add_argument('plot_date_in', type=str,
                        help='start date from which to extract')
    parser.add_argument('plot_date_fin', type=str,
                        help='end date until which to extract')
    parser.add_argument('time_res_plot', type=str,
                        help='Time Resolution of Data')
    parser.add_argument('path_to_in_mod_ts', type=str,
                        help='Path to input model extracted data')
    parser.add_argument('path_to_in_obs_ts', type=str,
                        help='Path to input observation extracted data')
    parser.add_argument('label_plot', type=str,
                        help='Name of Experiments in plots')
    parser.add_argument('time_res_axis', type=str,
                        help='Time Resolution of axis in the plots')
    parser.add_argument('path_to_output_exp', type=str,
                        help='Path to the output plot folder')
    parser.add_argument('path_to_comparison', type=str,
                        help='Path to the output plot folder for comparison')
    parser.add_argument('path_to_input_metadata_obs_file', type=str,
                        help='Path to accepted medatada observation file')
    args = parser.parse_args()
    return args


def main(args):
    date_in = args.plot_date_in
    date_fin = args.plot_date_fin
    time_res_arr = args.time_res_plot.split()
    path_to_mod_ts_arr = args.path_to_in_mod_ts.split()
    path_to_obs_ts = args.path_to_in_obs_ts
    label_plot_arr = args.label_plot.split()
    time_res_axis = args.time_res_axis
    path_to_output_experiments_arr = args.path_to_output_exp.split()
    path_to_comparison = args.path_to_comparison
    path_to_accepted_metadata_obs_file = args.path_to_input_metadata_obs_file

    start_date = date(int(date_in[0:4]), int(date_in[4:6]), int(date_in[6:8]))
    end_date = date(int(date_fin[0:4]), int(date_fin[4:6]), int(date_fin[6:8]))


    obs_file = {}

    with open(path_to_accepted_metadata_obs_file) as f:
        reader = csv.reader(f, delimiter=';')
        next(reader)
        for row in reader:
            # print(row)
            array_name = np.array([row[4], row[5], row[6]])
            boolArr = np.where(array_name != "_")
            obs_file[array_name[boolArr][0]] = {
                'lat': row[0], 'lon': row[1], 'depth': '3'}

    os.makedirs(path_to_comparison,  exist_ok=True)

    lon = {}
    lat = {}
    
    mapping(obs_file, lat, lon, path_to_comparison)
    
    sdev = {}
    crmsd = {}
    ccoef = {}
    vel_mod_ts = {}
    direction_mod_ts = {}
    
    for i, mod_folder in enumerate(path_to_mod_ts_arr):
        if time_res_arr[i][1] == 'M':
            timerange = pd.date_range(
                start_date, end_date, freq=time_res_arr[i][1]) - pd.DateOffset(days=15)
        elif time_res_arr[i][1] == 'D':
            timerange = pd.date_range(
                start_date, end_date, freq=time_res_arr[i][1]) + pd.DateOffset(hours=12)

        onlyfiles_mod = [f for f in sorted(
            listdir(mod_folder)) if isfile(join(mod_folder, f))]
        onlyfiles_obs = [f for f in sorted(
            listdir(path_to_obs_ts)) if isfile(join(path_to_obs_ts, f))]

        os.makedirs(path_to_output_experiments_arr[i],  exist_ok=True)

        vel_mod_ts[i] = {}
        direction_mod_ts[i] = {}
        vel_obs_ts = {}
        direction_obs_ts = {}
        
        for filename_mod, filename_obs in zip(onlyfiles_mod, onlyfiles_obs):
            
            splitted_name = np.array(filename_mod.split("_"))
            start_date_index = np.argwhere(splitted_name == date_in)
            name_station_splitted = splitted_name[0:start_date_index[0][0]]
            name_station = '_'.join(name_station_splitted)
            
            mod_ts = xr.open_dataset(mod_folder + filename_mod)
            mod_ts = mod_ts.resample(TIME=time_res_arr[i]).mean(skipna=True)
            
            vel_mod_ts[i][name_station] = mod_ts['mod_vel'].data[:]
            direction_mod_ts[i][name_station] = mod_ts['mod_direction'].data[:]
            
            obs_ts = xr.open_dataset(path_to_obs_ts + filename_obs)
            obs_ts = obs_ts.resample(TIME=time_res_arr[i]).mean(skipna=True)
            
            vel_obs_ts[name_station] = obs_ts['vel'].data[:]
            direction_obs_ts[name_station] = obs_ts['dir'].data[:]

    min_mooring_vel = {}
    max_mooring_vel = {}
    ymin = {}
    ymax = {}

    for key_obs_file, name_stat in zip(sorted(obs_file.keys()), vel_mod_ts[0].keys()):
        ymin[name_stat] = {}
        ymax[name_stat] = {}

        plot_mod_obs_ts_diff_comparison(name_stat, date_in, date_fin, time_res_arr, obs_file, key_obs_file,
                                        vel_mod_ts, vel_obs_ts, label_plot_arr, timerange, time_res_axis, path_to_comparison)
        plot_mod_obs_ts_comparison(name_stat, date_in, date_fin, time_res_arr, obs_file, key_obs_file,
                                   vel_mod_ts, vel_obs_ts, label_plot_arr, timerange, time_res_axis, path_to_comparison)
        ii = np.isfinite(np.array(vel_obs_ts[name_stat]))
        plot_mod_obs_ECDF_comparison(ii, name_stat, date_in, date_fin, time_res_arr, obs_file, key_obs_file,
                                     vel_mod_ts, vel_obs_ts, label_plot_arr, path_to_comparison, '_mod_obs_ECDF_comparison.png')

        name_file_substring = "_" + name_stat + "_windrose"
        title_substring = "Windrose " + name_stat
        a = direction_obs_ts[name_stat]
        b = vel_obs_ts[name_stat]
        a = a[~np.isnan(b)]
        b = b[~np.isnan(b)]

        min_mod_mooring_vel = np.inf
        max_mod_mooring_vel = 0

        for i in range(len(label_plot_arr)):
            min_vel_dict = np.nanmin(
                vel_mod_ts[i][name_stat][~np.isnan(vel_obs_ts[name_stat])][~np.isnan(a)])
            min_mod_mooring_vel = min(min_mod_mooring_vel, min_vel_dict)
            max_vel_dict = np.nanmax(
                vel_mod_ts[i][name_stat][~np.isnan(vel_obs_ts[name_stat])][~np.isnan(a)])
            max_mod_mooring_vel = max(max_mod_mooring_vel, max_vel_dict)

        min_mooring_vel[name_stat] = min(
            np.nanmin(b[~np.isnan(a)]), min_mod_mooring_vel)
        max_mooring_vel[name_stat] = max(
            np.nanmax(b[~np.isnan(a)]), max_mod_mooring_vel)

        ax = WindroseAxes.from_ax()
        turbo = plt.get_cmap('turbo')
        ax.bar(a[~np.isnan(a)], b[~np.isnan(a)], normed=True, bins=np.linspace(
            min_mooring_vel[name_stat], max_mooring_vel[name_stat], 5), opening=0.8, edgecolor='white', cmap=turbo)
        ymin[name_stat]["mooring"], ymax[name_stat]["mooring"] = ax.get_ylim()
        #plot_windrose(a[~np.isnan(a)],b[~np.isnan(a)],min_mooring_vel[name_stat], max_mooring_vel[name_stat],date_in,date_fin,name_file_substring,title_substring,path_to_comparison)

    plot_tot_mod_obs_ECDF_comparison(date_in, date_fin, time_res_arr, vel_mod_ts,
                                     vel_obs_ts, label_plot_arr, path_to_comparison, '_mod_obs_ECDF_comparison.png')

    bias_ts = {}
    diff_q_ts = {}
    rmsd_ts = {}

    for exp in range(len(label_plot_arr)):
        bias_ts[exp] = {}
        diff_q_ts[exp] = {}
        rmsd_ts[exp] = {}

        for instant in timerange:
            if time_res_arr[exp][1] == 'D':
                timetag = instant.strftime("%Y%m%d")
            if time_res_arr[exp][1] == 'M':
                timetag = instant.strftime("%Y%m")
            bias_ts[exp][timetag] = 0
            diff_q_ts[exp][timetag] = 0
            rmsd_ts[exp][timetag] = 0

        for i, instant in enumerate(timerange):
            #timetag = day.strftime("%Y%m%d")
            # print(timetag)
            if time_res_arr[exp][1] == 'D':
                timetag = instant.strftime("%Y%m%d")
            if time_res_arr[exp][1] == 'M':
                timetag = instant.strftime("%Y%m")
            for name_stat in vel_mod_ts[0].keys():
                if np.isnan(vel_obs_ts[name_stat][i]) or np.isnan(vel_mod_ts[exp][name_stat][i]):
                    print("nan beccato!")
                    continue
                else:
                    bias = vel_mod_ts[exp][name_stat][i] - \
                        vel_obs_ts[name_stat][i]

                    diff_q = (vel_mod_ts[exp][name_stat]
                              [i] - vel_obs_ts[name_stat][i])**2

                    bias_ts[exp][timetag] = bias_ts[exp][timetag] + bias
                    diff_q_ts[exp][timetag] = diff_q_ts[exp][timetag] + diff_q

            rmsd_ts[exp][timetag] = diff_q_ts[exp][timetag] / \
                len(vel_mod_ts[exp].keys())

            bias_ts[exp][timetag] = bias_ts[exp][timetag] / \
                len(vel_mod_ts[exp].keys())

    plot_bias_ts_comparison(date_in, date_fin, time_res_arr, timerange,
                            label_plot_arr, bias_ts, time_res_axis, path_to_comparison)
    plot_rmse_ts_comparison(date_in, date_fin, time_res_arr, timerange,
                            label_plot_arr, rmsd_ts, time_res_axis, path_to_comparison)
    #plot_bias_rmse_ts(date_in, date_fin, time_res, timerange, bias_ts, rmsd_ts, statistics_array, time_res_xaxis, name_exp, path_to_output_plot_folder)

    label_for_taylor = list(
        np.append('Non-Dimensional Observation', label_plot_arr))
    markers = ['o', 's', '^', '+', 'x', 'D']

    for key_obs_file, name_stat in zip(sorted(obs_file.keys()), vel_mod_ts[0].keys()):
        for exp in range(len(label_plot_arr)):
            a = vel_mod_ts[exp][name_stat]
            b = vel_obs_ts[name_stat]
            a = a[~np.isnan(b)]
            b = b[~np.isnan(b)]

            taylor_stats = sm.taylor_statistics(
                a[~np.isnan(a)], b[~np.isnan(a)])

            if exp == 0:
                sdev[name_stat] = list(
                    np.around(np.array([taylor_stats['sdev'][0], taylor_stats['sdev'][1]]), 4))
                print("sdev: ", taylor_stats['sdev'][0])
                print("sdev: ", taylor_stats['sdev'][1])
                crmsd[name_stat] = list(
                    np.around(np.array([taylor_stats['crmsd'][0], taylor_stats['crmsd'][1]]), 4))
                print("crmsd: ", taylor_stats['crmsd'][0])
                print("crmsd: ", taylor_stats['crmsd'][1])
                ccoef[name_stat] = list(
                    np.around(np.array([taylor_stats['ccoef'][0], taylor_stats['ccoef'][1]]), 4))
                print("ccoef: ", taylor_stats['ccoef'])
                print("ccoef: ", taylor_stats['ccoef'][0])
                print("ccoef: ", taylor_stats['ccoef'][1])
            else:
                append_value(sdev, name_stat, round(
                    taylor_stats['sdev'][1], 4))
                print("sdev: ", taylor_stats['sdev'][1])
                append_value(crmsd, name_stat, round(
                    taylor_stats['crmsd'][1], 4))
                print("crmsd: ", taylor_stats['crmsd'][1])
                append_value(ccoef, name_stat, round(
                    taylor_stats['ccoef'][1], 4))
                print("ccoef: ", taylor_stats['ccoef'])
                print("ccoef: ", taylor_stats['ccoef'][1])

        obsSTD = [sdev[name_stat][0]]
        s = sdev[name_stat][1:]
        r = ccoef[name_stat][1:]

        l = label_for_taylor[1:]

        fname = name_stat + '_TaylorDiagram.png'
        srl(obsSTD, s, r, l, fname, markers, path_to_comparison)

    statistics = {}
    qq_statistics = {}
    possible_colors = ['red', 'blue', 'black', 'green',
                       'purple', 'orange', 'brown', 'pink', 'grey', 'olive']
    possible_markers = np.array(["o", "^", "s", "P", "*", "+"])
    string_time_res = Get_String_Time_Resolution(
        start_date, end_date, time_res_arr[0])
    for exp in range(len(label_plot_arr)):
        statistics[exp] = {}
        qq_statistics[exp] = {}
        len_not_nan_values = []
        moor_names = []
        for counter, (key_obs_file, name_stat) in enumerate(zip(sorted(obs_file.keys()), vel_mod_ts[exp].keys())):

            plotname = label_plot_arr[exp] + '_' + name_stat + '_' + \
                date_in + '_' + date_fin + '_' + \
                time_res_arr[exp] + '_scatterPlot.png'
            title = 'Surface (3m) Current Velocity ' + name_stat + '\n (' + \
                obs_file[key_obs_file]['lat'] + ', ' + obs_file[key_obs_file]['lon'] + \
                    ') Period: ' + date_in + ' - ' + date_fin
            xlabel = 'Observation Current Velocity [m/s]'
            ylabel = 'Model Current Velocity [m/s]'
            
            moor_names.append(name_stat)
            ciao = np.array(vel_obs_ts[name_stat])
            len_not_nan_values.append(len(ciao[~np.isnan(ciao)]))
            statistics_array = scatterPlot(np.array(vel_mod_ts[exp][name_stat]), np.array(vel_obs_ts[name_stat]), path_to_output_experiments_arr[exp] + plotname, label_plot_arr[exp], 1, len(
                vel_mod_ts[exp][name_stat]), possible_markers[counter], name_stat, possible_colors, string_time_res, title=title, xlabel=xlabel, ylabel=ylabel)
            mean_vel_mod = round(np.nanmean(
                np.array(vel_mod_ts[exp][name_stat])), 2)
            mean_vel_obs = round(np.nanmean(
                np.array(vel_obs_ts[name_stat])), 2)
            tot_mean_stat = [mean_vel_mod, mean_vel_obs]
            row_stat = tot_mean_stat + statistics_array
            statistics[exp][name_stat] = row_stat

            plotname = label_plot_arr[exp] + '_' + name_stat + '_' + \
                date_in + '_' + date_fin + '_' + \
                time_res_arr[exp] + '_qqPlot.png'
            title = 'Surface (3m) Current Velocity ' + name_stat + '\n (' + \
                obs_file[key_obs_file]['lat'] + ', ' + obs_file[key_obs_file]['lon'] + \
                    ') Period: ' + date_in + ' - ' + date_fin
            xlabel = 'Observation Current Velocity [m/s]'
            ylabel = 'Model Current Velocity [m/s]'
            qq_statistics_array = QQPlot(np.array(vel_mod_ts[exp][name_stat]), np.array(
                vel_obs_ts[name_stat]), path_to_output_experiments_arr[exp] + plotname, label_plot_arr[exp], title=title, xlabel=xlabel, ylabel=ylabel)
            mean_vel_mod = round(np.nanmean(
                np.array(vel_mod_ts[exp][name_stat])), 2)
            mean_vel_obs = round(np.nanmean(
                np.array(vel_obs_ts[name_stat])), 2)
            tot_mean_stat = [mean_vel_mod, mean_vel_obs]
            row_stat = tot_mean_stat + qq_statistics_array
            qq_statistics[exp][name_stat] = row_stat

            name_file_substring = "_" + \
                label_plot_arr[exp] + "_" + name_stat + "_windrose"
            title_substring = label_plot_arr[exp] + " Windrose " + name_stat
            a = direction_mod_ts[exp][name_stat]
            b = vel_mod_ts[exp][name_stat]

            c = direction_obs_ts[name_stat]
            d = vel_obs_ts[name_stat]

            mask = np.isreal(a) & np.isreal(b) & np.isreal(c) & np.isreal(
                d) & ~np.isnan(a) & ~np.isnan(b) & ~np.isnan(c) & ~np.isnan(d)

            ax = WindroseAxes.from_ax()
            turbo = plt.get_cmap('turbo')
            ax.bar(a[mask], b[mask], normed=True, bins=np.linspace(min_mooring_vel[name_stat],
                   max_mooring_vel[name_stat], 5), opening=0.8, edgecolor='white', cmap=turbo)
            ymin[name_stat][exp], ymax[name_stat][exp] = ax.get_ylim()

            #plot_windrose(a[mask],b[mask],min_mooring_vel[name_stat], max_mooring_vel[name_stat],date_in,date_fin,name_file_substring,title_substring,path_to_output_experiments_arr[exp])

            mod_array = np.array([])
            obs_array = np.array([])
            for name_stat in vel_mod_ts[exp].keys():
                mod_array = np.concatenate(
                    [mod_array, np.array(vel_mod_ts[exp][name_stat])])
                obs_array = np.concatenate(
                    [obs_array, np.array(vel_obs_ts[name_stat])])

        tot_mean_mod = round(np.nanmean(mod_array), 2)
        tot_mean_obs = round(np.nanmean(obs_array), 2)
        mean_all = [tot_mean_mod, tot_mean_obs]

        plotname = label_plot_arr[exp] + '_' + date_in + '_' + \
            date_fin + '_' + time_res_arr[exp] + '_scatterPlot.png'
        title = 'Surface (3m) Current Velocity -ALL \n Period: ' + \
            date_in + '-' + date_fin
        xlabel = 'Observation Current Velocity [m/s]'
        ylabel = 'Model Current Velocity [m/s]'
        statistics_array = scatterPlot(mod_array, obs_array, path_to_output_experiments_arr[exp] + plotname, label_plot_arr[exp], len(
            onlyfiles_mod), timerange.shape[0], possible_markers, moor_names, possible_colors, string_time_res, len_not_nan_values=len_not_nan_values, title=title, xlabel=xlabel, ylabel=ylabel)
        row_all = mean_all + statistics_array
        statistics[exp]["ALL BUOYS"] = row_all

        plotname = label_plot_arr[exp] + '_' + date_in + '_' + \
            date_fin + '_' + time_res_arr[exp] + '_qqPlot.png'
        title = 'Surface (3m) Current Velocity -ALL \n Period: ' + \
            date_in + '-' + date_fin
        xlabel = 'Observation Current Velocity [m/s]'
        ylabel = 'Model Current Velocity [m/s]'
        qq_statistics_array = QQPlot(
            mod_array, obs_array, path_to_output_experiments_arr[exp] + plotname, label_plot_arr[exp], title=title, xlabel=xlabel, ylabel=ylabel)
        row_all = mean_all + statistics_array
        qq_statistics[exp]["ALL BUOYS"] = row_all

        a_file = open(path_to_output_experiments_arr[exp] + "statistics_" +
                      label_plot_arr[exp] + "_" + date_in + "_" + date_fin + ".csv", "w")
        writer = csv.writer(a_file)
        writer.writerow(["name_station", "mean_mod", "mean_obs",
                        "bias", "rmse", "si", "corr", "stderr", "number_of_obs"])
        for key, value in statistics[exp].items():
            array = [key] + value
            print(array)
            writer.writerow(array)
        a_file.close()

        a_file = open(path_to_output_experiments_arr[exp] + "qq_statistics_" +
                      label_plot_arr[exp] + "_" + date_in + "_" + date_fin + ".csv", "w")
        writer = csv.writer(a_file)
        writer.writerow(["name_station", "mean_mod", "mean_obs",
                        "bias", "rmse", "si", "corr", "stderr", "number_of_obs"])
        for key, value in qq_statistics[exp].items():
            array = [key] + value
            print(array)
            writer.writerow(array)
        a_file.close()

    for key_obs_file, name_stat in zip(sorted(obs_file.keys()), vel_mod_ts[0].keys()):
        name_file_substring = "_" + name_stat + "_windrose"
        title_substring = "Windrose " + name_stat
        a = direction_obs_ts[name_stat]
        b = vel_obs_ts[name_stat]
        a = a[~np.isnan(b)]
        b = b[~np.isnan(b)]
        
        min_value = min(ymin[name_stat].values())
        min_key = [k for k, v in ymin[name_stat].items() if v == min_value][0]
        max_value = max(ymax[name_stat].values())
        max_key = [k for k, v in ymax[name_stat].items() if v == max_value][0]
        
        plot_windrose(a[~np.isnan(a)], b[~np.isnan(a)], min_mooring_vel[name_stat], max_mooring_vel[name_stat], date_in, date_fin,
                      name_file_substring, title_substring, path_to_comparison, ymin[name_stat][min_key], ymax[name_stat][max_key])

        for exp in range(len(label_plot_arr)):

            name_file_substring = "_" + \
                label_plot_arr[exp] + "_" + name_stat + "_windrose"
            title_substring = label_plot_arr[exp] + " Windrose " + name_stat
            a = direction_mod_ts[exp][name_stat]
            b = vel_mod_ts[exp][name_stat]

            c = direction_obs_ts[name_stat]
            d = vel_obs_ts[name_stat]
            mask = np.isreal(a) & np.isreal(b) & np.isreal(c) & np.isreal(
                d) & ~np.isnan(a) & ~np.isnan(b) & ~np.isnan(c) & ~np.isnan(d)

            plot_windrose(a[mask], b[mask], min_mooring_vel[name_stat], max_mooring_vel[name_stat], date_in, date_fin, name_file_substring,
                          title_substring, path_to_output_experiments_arr[exp], ymin[name_stat][min_key], ymax[name_stat][max_key])


if __name__ == "__main__":

    args = parse_args()
    main(args)
