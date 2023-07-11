from core import Filter_Data, Mask_For_Depth, Check_Quality_and_Filter, Check_Nan, Remove_Outliers
import matplotlib.dates
import csv
from datetime import timedelta
import xarray as xr
import numpy as np
import pandas as pd
import math
from datetime import datetime
from os.path import isfile, join
from os import listdir
import os.path
import warnings
import argparse
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")


def daterange(start_date, end_date):

    for n in range(int((end_date - start_date).days)+1):
        yield start_date + timedelta(n)


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


def parse_args():
    parser = argparse.ArgumentParser(
        description='Extraction of Obs Time Series')
    parser.add_argument('date_in', type=str,
                        help='start date from which to extract')
    parser.add_argument('date_fin', type=str,
                        help='end date until which to extract')
    parser.add_argument('path_to_metadata_obs_file', type=str,
                        help='Path to the metadata file of available observations')
    parser.add_argument('time_res_to_average', type=str,
                        help='Time resolution to average the obs data')
    parser.add_argument('depth_obs', type=float,
                        help='Depth at (or around) which observations are extracted')
    parser.add_argument('nan_treshold', type=float,
                        help='Description of nan_treshold argument')
    parser.add_argument('path_to_accepted_metadata_obs_file', type=str,
                        help='Path to the metadata file of accepted observations')
    parser.add_argument('path_to_out_obs_ts', type=str,
                        help='Path to the output folder')
    parser.add_argument('path_to_plot_ts', type=str,
                        help='Path to the output plot folder')
    args = parser.parse_args()
    return args


def main(args):
    # Extract values from command line arguments
    date_in = args.date_in
    date_fin = args.date_fin
    path_to_metadata_obs_file = args.path_to_metadata_obs_file
    time_res_to_average = args.time_res_to_average
    depth_obs = args.depth_obs
    nan_treshold = args.nan_treshold
    path_to_accepted_metadata_obs_file = args.path_to_accepted_metadata_obs_file
    path_to_out_obs_ts = args.path_to_out_obs_ts
    path_to_plot_ts = args.path_to_plot_ts

    os.makedirs(path_to_out_obs_ts, exist_ok=True)
    os.makedirs(path_to_plot_ts, exist_ok=True)

    filtered_data, removed_data = Filter_Data(
        date_in, date_fin, path_to_metadata_obs_file, path_to_accepted_metadata_obs_file)

    with open(path_to_accepted_metadata_obs_file, 'r') as file:
        header_row = file.readline().rstrip()

        # remove the semicolon at the end of the header row, if present
        if header_row[-1] == ';':
            header_row = header_row[:-1]

        column_names = header_row.split(';')
    # read the CSV file again and skip the first row (header)
    df = pd.read_csv(path_to_accepted_metadata_obs_file, sep=';')
    qf_value = column_names[9]
    path_to_obs_file = column_names[10]
    name = column_names[4]
    CMEMS_code = column_names[5]
    WMO_code = column_names[6]

    dict_speed = {}
    for index, row in df.iterrows():
        ds = xr.open_dataset(df[path_to_obs_file][index])
        ds_restricted = ds.sel(TIME=slice(date_in, date_fin))
        mask_depth = Mask_For_Depth(ds_restricted)
        times, speed_values, dir_values, EWCT_values, NSCT_values = Check_Quality_and_Filter(
            ds_restricted, df, qf_value, mask_depth)

        # Choose between 'name', 'CMEMS code', and 'WMO code' as per your requirements
        station_info = df.iloc[index][[name, CMEMS_code, WMO_code]].str.strip()
        # select elements with letters
        selected_name = station_info.loc[station_info.str.contains(
            '[a-zA-Z]', na=False, regex=True)]

        # if no elements with letters, select elements with numbers
        if selected_name.empty:
            selected_name_1 = station_info.str.extract(
                r'(\d+)').astype(float).values
            selected_name_1 = [
                item for sublist in selected_name_1 for item in sublist]
            for item in selected_name_1:
                if math.isnan(item) == False:
                    selected_name_end = item
        else:
            selected_name_end = selected_name.values[0]

        dict_speed[selected_name_end] = {
            'datetime': times, 'velocity': speed_values, 'direction': dir_values, 'EWCT': EWCT_values, 'NSCT': NSCT_values}

    removed_outliers_dict = Remove_Outliers(dict_speed)

    daily_dict = {}
    end_filtered_df = pd.DataFrame(columns=column_names)
    for count, (dict_key, dict_value) in enumerate(removed_outliers_dict.items()):
        # set the datetime column as the index of the DataFrame
        if np.size(dict_value['EWCT'][:]) > 0 and np.size(dict_value['NSCT'][:]) > 0:
            df_rem_out_speed = pd.DataFrame(
                {'datetime': dict_value['datetime'][:], 'velocity': dict_value['velocity'][:], 'direction': dict_speed[dict_key]['direction'][:], 'EWCT': dict_value['EWCT'][:], 'NSCT': dict_value['NSCT'][:]})
        else:
            df_rem_out_speed = pd.DataFrame(
                {'datetime': dict_value['datetime'][:], 'velocity': dict_value['velocity'][:], 'direction': dict_speed[dict_key]['direction'][:]})
        df_rem_out_speed.set_index('datetime', inplace=True)

        if len(dict_value['datetime'][:]) > 0:
            if not isinstance(df_rem_out_speed.index, pd.DatetimeIndex):
                df_rem_out_speed.index = pd.to_datetime(df_rem_out_speed.index)
        else:
            continue

        # resample the DataFrame by day, calculating the mean of each group
        df_rem_out_speed_resampled = df_rem_out_speed.resample(
            time_res_to_average).mean()

        # retrieve the velocity column from the resampled DataFrame
        daily_velocities = df_rem_out_speed_resampled['velocity'].tolist()
        daily_directions = df_rem_out_speed_resampled['direction'].tolist()
        if np.size(dict_value['EWCT'][:]) > 0 and np.size(dict_value['NSCT'][:]) > 0:
            daily_EWCT = df_rem_out_speed_resampled['EWCT'].tolist()
            daily_NSCT = df_rem_out_speed_resampled['NSCT'].tolist()
        daily_times = df_rem_out_speed_resampled.index.tolist()

        if not np.isnan(daily_velocities).all():
            bool_nan = Check_Nan(daily_times, daily_velocities, nan_treshold)
            if bool_nan:
                if np.size(dict_value['EWCT'][:]) > 0 and np.size(dict_value['NSCT'][:]) > 0:
                    print("sono nell if")
                    print(daily_EWCT)
                    print(daily_NSCT)
                    daily_dict[dict_key] = {
                        'datetime': daily_times, 'velocity': daily_velocities, 'direction': daily_directions, 'EWCT': daily_EWCT, 'NSCT': daily_NSCT}
                else:
                    daily_dict[dict_key] = {
                        'datetime': daily_times, 'velocity': daily_velocities, 'direction': daily_directions}
                end_filtered_df.loc[count] = df.loc[count]

    end_filtered_df.to_csv(
        path_to_accepted_metadata_obs_file, sep=';', index=False)

    for dict_key, daily_dict_value in daily_dict.items():
        fig, (ax1, ax2, ax3) = plt.subplots(3)
        fig.subplots_adjust(hspace=1)
        fig.suptitle('Mooring: {}'.format(dict_key))
        max_value = max(dict_speed[dict_key]['velocity'][:])
        min_value = min(dict_speed[dict_key]['velocity'][:])
        ax1.plot(dict_speed[dict_key]['datetime'][:],
                 dict_speed[dict_key]['velocity'][:])
        ax1.tick_params(axis='x', rotation=45, labelsize=7)
        ax1.set_title('Original Velocity Time Series')
        ax1.grid()
        ax2.plot(removed_outliers_dict[dict_key]['datetime']
                 [:], removed_outliers_dict[dict_key]['velocity'][:])
        ax2.tick_params(axis='x', rotation=45, labelsize=7)
        ax2.set_title('Velocity Time Series (no outliers)')
        ax2.grid()
        ax3.plot(daily_dict_value['datetime'][:],
                 daily_dict_value['velocity'][:])
        ax3.tick_params(axis='x', rotation=45, labelsize=7)
        ax3.set_title('Averaged velocity timeseries (no outliers)')
        ax3.grid()
        # Optional padding
        padding = 0.01  # 10% padding

        # Calculate the y-axis limits
        y_min = min_value - padding
        y_max = max_value + padding
        ax1.set_ylim(y_min, y_max)
        ax2.set_ylim(y_min, y_max)
        ax3.set_ylim(y_min, y_max)
        filename = dict_key + '_plot_speed_ts.png'
        fig.savefig(path_to_plot_ts + filename)

        fig_dir, (ax1_dir, ax2_dir) = plt.subplots(2)
        fig_dir.subplots_adjust(hspace=1)
        fig_dir.suptitle('Mooring: {}'.format(dict_key))
        ax1_dir.plot(dict_speed[dict_key]['datetime'][:],
                     dict_speed[dict_key]['direction'][:])
        ax1_dir.tick_params(axis='x', rotation=45, labelsize=7)
        ax1_dir.set_title('Original Direction Time Series')
        ax1_dir.grid()
        ax2_dir.plot(daily_dict_value['datetime']
                     [:], daily_dict_value['direction'][:])
        ax2_dir.tick_params(axis='x', rotation=45, labelsize=7)
        ax2_dir.set_title('Averaged Direction Time Series')
        ax2_dir.grid()
        filename = dict_key + '_plot_dir_ts.png'
        fig_dir.savefig(path_to_plot_ts + filename)

        if "EWCT" in daily_dict_value:
            print("if pt 2")
            obs_ds = xr.Dataset(data_vars=dict(TIME=(['TIME'], daily_dict_value['datetime'][:]), vel=(
                ['TIME'], daily_dict_value['velocity'][:]), dir=(
                ['TIME'], daily_dict_value['direction'][:]), EWCT=(
                ['TIME'], daily_dict_value['EWCT'][:]), NSCT=(
                ['TIME'], daily_dict_value['NSCT'][:])))
        else:
            obs_ds = xr.Dataset(data_vars=dict(TIME=(['TIME'], daily_dict_value['datetime'][:]), vel=(
                ['TIME'], daily_dict_value['velocity'][:]), dir=(
                ['TIME'], daily_dict_value['direction'][:])))
        output_obs_file = dict_key + "_" + date_in + "_" + date_fin + "_obs.nc"

        obs_ds.to_netcdf(path=path_to_out_obs_ts + output_obs_file)
        #c_direction=(['TIME'], dict_daily_c_dir_cardinal[count][:])


if __name__ == "__main__":

    args = parse_args()
    main(args)
