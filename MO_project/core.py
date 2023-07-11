import os
import datetime
import glob
import pandas as pd
import csv
import numpy as np
from datetime import timedelta, date
from netCDF4 import Dataset
import xarray as xr
import sys
from scipy.spatial import cKDTree
import math


def Link_Files(paths, old_names, new_names, date_in, date_fin, time_res, out_paths):
    current_date = date_in
    for path in out_paths:
        os.makedirs(path, exist_ok=True)
    # Remove all files in the directory if it already exists
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

    while current_date <= date_fin:
        print(current_date)
        # Loop through paths
        for i, path in enumerate(paths):
            # Find files
            # for root, dirs, files in os.walk(path):
            for file in glob.glob(path + "/" + current_date.strftime("%Y%m%d") + "/model/*"):
                # for file in files:
                u_file = None
                v_file = None
                if old_names[i] in file and time_res[i] in file and current_date.strftime('%Y%m%d') in file:
                    print(file)
                    if 'U' in file:
                        #u_file = os.path.join(root, file)
                        u_file = file
                    elif 'V' in file:
                        #v_file = os.path.join(root, file)
                        v_file = file

                # Link files to output folder
                    if u_file is not None and v_file is not None:
                        os.symlink(u_file, os.path.join(
                            out_paths[i], f"{new_names[i]}_{time_res[i]}_{current_date.strftime('%Y%m%d')}_grid_U.nc"))
                        os.symlink(v_file, os.path.join(
                            out_paths[i], f"{new_names[i]}_{time_res[i]}_{current_date.strftime('%Y%m%d')}_grid_V.nc"))
                    elif u_file is not None and 'U' in u_file:
                        os.symlink(u_file, os.path.join(
                            out_paths[i], f"{new_names[i]}_{time_res[i]}_{current_date.strftime('%Y%m%d')}_grid_U.nc"))
                    elif v_file is not None and 'V' in v_file:
                        os.symlink(v_file, os.path.join(
                            out_paths[i], f"{new_names[i]}_{time_res[i]}_{current_date.strftime('%Y%m%d')}_grid_V.nc"))

        # Increment date
        current_date += datetime.timedelta(days=1)


def wind_direction(x, y):

    # Compute the angle in radians
    angle_rad = np.arctan2(y, x)

    # Convert the angle to degrees
    angle_deg = np.degrees(angle_rad)

    # Adjust the angle so that east is 0 degrees
    # and angles increase counterclockwise
    angle_deg = 90 - angle_deg

    # Ensure the angle is in the range [0, 360)
    angle_deg = angle_deg % 360
    return angle_deg


def Filter_Data(start_date, end_date, obs_file_path, path_to_accepted_metadata_obs_file):
    # read data from csv
    with open(obs_file_path) as f:
        first_line = f.readline()
        column_names = first_line.replace("#", "")
        column_names = column_names.split(';')

        reader = csv.reader(
            filter(lambda row: row[0] != '#', f), delimiter=';')

        # use column names from file
        data = pd.DataFrame(reader, columns=column_names)

    last_lat = column_names[0]
    last_lon = column_names[1]
    num_vlevs = column_names[2]
    num_sfc_levs = column_names[3]
    name = column_names[4]
    CMEMS_code = column_names[5]
    WMO_code = column_names[6]
    time_period = column_names[7]
    fields_list = column_names[8]
    qf_value = column_names[9]
    path_to_obs_file = column_names[10]

    # apply filters

    mask_time_period = (data[time_period].apply(lambda x: pd.to_datetime(datetime.datetime.strptime(x.split(',')[0], "%Y-%m-%d"))) <= start_date) & \
        (data[time_period].apply(lambda x: pd.to_datetime(
            datetime.datetime.strptime(x.split(',')[1], "%Y-%m-%d"))) >= end_date)

    mask_field_list = (data[fields_list].str.contains('HCSP')) | \
        (data[fields_list].str.contains('EWCT') &
         data[fields_list].str.contains('NSCT'))

    mask_num_sfc_levs = data[num_sfc_levs].astype(int) != 0

    filtered_data = data.loc[mask_time_period &
                             mask_field_list & mask_num_sfc_levs]

    # get removed rows and reason for removal
    removed_data_mask = ~(
        mask_time_period & mask_field_list & mask_num_sfc_levs)
    removed_data = data.loc[removed_data_mask]
    removed_data_reasons = []

    for i, row in removed_data.iterrows():
        reasons = []
        if not mask_time_period[i]:
            reasons.append('time period')
        if not mask_field_list[i]:
            reasons.append('field list')
        if not mask_num_sfc_levs[i]:
            reasons.append('num sfc vlevs')

        # Choose between 'name', 'CMEMS code', and 'WMO code' as per your requirements
        station_info = row[[name, CMEMS_code, WMO_code]].str.strip()
        # select elements with letters
        selected_name = station_info.loc[station_info.str.contains('[a-zA-Z]')]

        # if no elements with letters, select elements with numbers
        if selected_name.empty:
            selected_name = station_info.str.extract(r'(\d+)').astype(float)
            selected_name = selected_name.loc[~selected_name.isna()]

        selected_name = selected_name.iloc[0]
        if len(reasons) == 1:
            reasons.append(f'not available for station: {selected_name}')

            # Add the station name to the reasons list

            removed_data_reasons.append(" ".join(reasons))
        else:
            reasons.append(f'not available for station: {selected_name}')

            # Add the station name to the reasons list

            removed_data_reasons.append(", ".join(reasons))

    # add reasons column to removed_data and save to csv
    removed_data = pd.DataFrame({})
    removed_data['removal_reasons'] = removed_data_reasons

    #removed_data = removed_data.apply(lambda x: x.str.strip('"'))
    removed_data.to_csv("./removed_data.csv", sep=';', index=False)

    # save filtered data to new csv file
    print(filtered_data)
    filtered_data.dropna(
        axis=0,
        how='any',
        inplace=True
    )
    print(filtered_data)
    filtered_data.to_csv(
        path_to_accepted_metadata_obs_file, sep=';', index=False)
    text = open(path_to_accepted_metadata_obs_file, "r")
    text = ''.join([i for i in text])
    # search and replace the contents
    text = text.replace('"', "")
    x = open(path_to_accepted_metadata_obs_file, "w")
    x.writelines(text)
    x.close()
    # print number of filtered and removed rows
    print(f'{len(filtered_data)} rows were kept.')
    print(f'{len(removed_data)} rows were removed.')

    return filtered_data, removed_data


def Mask_For_Depth(dataset):
    mask_depth_three = dataset['DEPH'] == 3
    print(np.count_nonzero(mask_depth_three))

    mask_range = (((dataset['DEPH'] >= 2) & (
        dataset['DEPH'] <= 4)) & (dataset['DEPH'] != 3))
    print(np.count_nonzero(mask_range))

    difference = abs(dataset['DEPH'] - 3)
    masked_difference_values = difference.where(mask_range)
    # Find the column index of the minimum value in each row
    # Replace all-NaN slices with np.inf
    masked_difference_values.data[np.isnan(
        masked_difference_values.data).all(axis=1)] = np.inf
    # Find the rows with all-np.inf values
    inf_rows = np.isinf(masked_difference_values.data).all(axis=1)

    min_col_indices = np.nanargmin(masked_difference_values.data, axis=1)

    # Find the row index for each minimum value
    row_indices = np.arange(masked_difference_values.shape[0])

    # Combine the row and column indices into a 2D array of coordinates
    coordinates = np.stack((row_indices, min_col_indices), axis=1)
    # Create a boolean mask with False values
    new_mask = np.zeros_like(dataset['DEPH'], dtype=bool)

    # Set the elements corresponding to the minimum values to True
    new_mask[coordinates[:, 0], coordinates[:, 1]] = True
    new_mask[inf_rows, :] = False

    # print(mask_variable)
    mask1_no_true = ~mask_depth_three.any(axis=1)

    mask_merged = mask_depth_three.copy()
    mask_merged[mask1_no_true] = new_mask[mask1_no_true]
    print(np.count_nonzero(mask_merged))
    return mask_merged


def Check_Quality_and_Filter(dataset, dataframe, quality_flag, mask_depth):
    check_vars = ['HCSP', 'EWCT', 'NSCT', 'HCSP_QC', 'EWCT_QC', 'NSCT_QC']
    # Create empty lists for the time and values
    times = []
    values = []
    dir_values = []
    values_EWCT = []
    values_NSCT = []
    if all([x in dataset.data_vars for x in check_vars]):
        mask_quality_hcsp = dataset["HCSP_QC"] == dataframe[quality_flag][5]
        mask_end_hcsp = mask_depth.where(mask_quality_hcsp, other=False)

        mask_quality_hcdt = dataset["HCDT_QC"] == dataframe[quality_flag][5]
        mask_end_hcdt = mask_depth.where(mask_quality_hcdt, other=False)

        mask_quality_ewct = dataset["EWCT_QC"] == dataframe[quality_flag][5]
        mask_end_ewct = mask_depth.where(mask_quality_ewct, other=False)

        mask_quality_nsct = dataset["NSCT_QC"] == dataframe[quality_flag][5]
        mask_end_nsct = mask_depth.where(mask_quality_nsct, other=False)

        hcsp_filtered = dataset['HCSP'].where(mask_end_hcsp, other=np.nan)
        hcdt_filtered = dataset['HCDT'].where(mask_end_hcdt, other=np.nan)
        ewct_filtered = dataset['EWCT'].where(mask_end_ewct, other=np.nan)
        nsct_filtered = dataset['NSCT'].where(mask_end_nsct, other=np.nan)

        # Loop over the rows of the DataArray
        for i in range(hcsp_filtered.shape[0]):
            # Get the non-NaN value for this row
            value_hcsp = hcsp_filtered[i].values[np.isfinite(
                hcsp_filtered[i].values)]
            # If there are no non-NaN values, append NaN to the values list
            if len(value_hcsp) == 0:
                value_ewct = ewct_filtered[i].values[np.isfinite(
                    ewct_filtered[i].values)]
                value_nsct = nsct_filtered[i].values[np.isfinite(
                    nsct_filtered[i].values)]
                if len(value_ewct) == 0 or len(value_nsct) == 0:
                    values.append(np.nan)
                    dir_values.append(np.nan)
                else:
                    values.append(
                        np.sqrt(value_ewct[0] ^ 2 + value_nsct[0] ^ 2))
                    dir_values.append(wind_direction(
                        value_ewct[0], value_nsct[0]))

            # Otherwise, append the non-NaN value to the values list
            else:
                values.append(value_hcsp[0])
                value_hcdt = hcdt_filtered[i].values[np.isfinite(
                    hcdt_filtered[i].values)]
                if len(value_hcdt) == 0:
                    dir_values.append(np.nan)
                else:
                    dir_values.append(value_hcdt[0])
            # Append the time for this row to the times list
            times.append(hcsp_filtered.TIME[i].values)
    elif all([x in dataset.data_vars for x in ['HCSP', 'HCSP_QC']]):
        mask_quality_hcsp = dataset["HCSP_QC"] == dataframe[quality_flag][5]
        mask_end_hcsp = mask_depth.where(mask_quality_hcsp, other=False)

        mask_quality_hcdt = dataset["HCDT_QC"] == dataframe[quality_flag][5]
        mask_end_hcdt = mask_depth.where(mask_quality_hcdt, other=False)

        hcsp_filtered = dataset['HCSP'].where(mask_end_hcsp, other=np.nan)
        hcdt_filtered = dataset['HCDT'].where(mask_end_hcdt, other=np.nan)

        # Loop over the rows of the DataArray
        for i in range(hcsp_filtered.shape[0]):
            # Get the non-NaN value for this row
            value_hcsp = hcsp_filtered[i].values[np.isfinite(
                hcsp_filtered[i].values)]
            # If there are no non-NaN values, append NaN to the values list
            if len(value_hcsp) == 0:
                values.append(np.nan)
                dir_values.append(np.nan)
            # Otherwise, append the non-NaN value to the values list
            else:
                values.append(value_hcsp[0])
                value_hcdt = hcdt_filtered[i].values[np.isfinite(
                    hcdt_filtered[i].values)]
                if len(value_hcdt) == 0:
                    dir_values.append(np.nan)
                else:
                    dir_values.append(value_hcdt[0])
            # Append the time for this row to the times list
            times.append(hcsp_filtered.TIME[i].values)
    elif all([x in dataset.data_vars for x in ['EWCT', 'EWCT_QC', 'NSCT', 'NSCT_QC']]):
        mask_quality_ewct = dataset["EWCT_QC"] == dataframe[quality_flag][0]
        mask_end_ewct = mask_depth.where(mask_quality_ewct, other=False)

        mask_quality_nsct = dataset["NSCT_QC"] == dataframe[quality_flag][0]
        mask_end_nsct = mask_depth.where(mask_quality_nsct, other=False)

        ewct_filtered = dataset['EWCT'].where(mask_end_ewct, other=np.nan)
        nsct_filtered = dataset['NSCT'].where(mask_end_nsct, other=np.nan)

        # Loop over the rows of the DataArray
        for i in range(ewct_filtered.shape[0]):
            # Get the non-NaN value for this row
            value_ewct = ewct_filtered[i].values[np.isfinite(
                ewct_filtered[i].values)]
            value_nsct = nsct_filtered[i].values[np.isfinite(
                nsct_filtered[i].values)]
            # If there are no non-NaN values, append NaN to the values list
            if len(value_ewct) == 0 or len(value_nsct) == 0:
                values.append(np.nan)
                values_EWCT.append(np.nan)
                values_NSCT.append(np.nan)
                dir_values.append(np.nan)
            # Otherwise, append the non-NaN value to the values list
            else:
                values.append(np.sqrt(value_ewct[0]**2 + value_nsct[0]**2))
                values_EWCT.append(value_ewct[0])
                values_NSCT.append(value_nsct[0])
                dir_values.append(wind_direction(value_ewct[0], value_nsct[0]))
            # Append the time for this row to the times list
            times.append(ewct_filtered.TIME[i].values)
    return times, values, dir_values, values_EWCT, values_NSCT


def Check_Nan(times, velocities, treshold):
    bool_value = True
    nan_counter = np.count_nonzero(np.isnan(velocities))/len(velocities)
    print("nan counter: ", nan_counter)
    if nan_counter > float(treshold):
        bool_value = False

    return bool_value


def Remove_Outliers(dict_speed):
    update_velocity = {}

    for dict_key, dict_value in dict_speed.items():
        # print(dict_key)
        mean_value = np.nanmean(dict_value['velocity'][:])
        sd_value = np.nanstd(dict_value['velocity'][:])

        mean_EWCT_value = np.nanmean(dict_value['EWCT'][:])
        sd_EWCT_value = np.nanstd(dict_value['EWCT'][:])

        mean_NSCT_value = np.nanmean(dict_value['NSCT'][:])
        sd_NSCT_value = np.nanstd(dict_value['NSCT'][:])
        #mean_dir_value = np.nanmean(dict_value['direction'][:])
        #sd_dir_value = np.nanstd(dict_value['direction'][:])
        update_velocity[dict_key] = {'datetime': dict_value['datetime'], 'velocity': np.where((dict_value['velocity'] > (mean_value+2*sd_value)) | (
            dict_value['velocity'] < (mean_value-2*sd_value)), np.nan, dict_value['velocity']), 'EWCT': np.where((dict_value['EWCT'] > (mean_EWCT_value+2*sd_EWCT_value)) | (
                dict_value['EWCT'] < (mean_EWCT_value-2*sd_EWCT_value)), np.nan, dict_value['EWCT']), 'NSCT': np.where((dict_value['NSCT'] > (mean_NSCT_value+2*sd_NSCT_value)) | (
                    dict_value['NSCT'] < (mean_NSCT_value-2*sd_NSCT_value)), np.nan, dict_value['NSCT'])}
        # , 'direction': np.where((dict_value['direction'] > (mean_dir_value+2*sd_dir_value)) | (dict_value['direction'] < (mean_dir_value-2*sd_dir_value)), np.nan, dict_value['direction'])

    return update_velocity


def daterange(start_date, end_date, resolution):
    if resolution == 'd':
        for n in range(int((end_date - start_date).days)+1):
            yield start_date + timedelta(n)
    elif resolution == 'h':
        for n in range(int((end_date - start_date).total_seconds()//3600)+1):
            yield start_date + timedelta(hours=n)
    elif resolution == 'm':
        for n in range(int((end_date - start_date).total_seconds()//60)+1):
            yield start_date + timedelta(minutes=n)
    elif resolution == 's':
        for n in range(int((end_date - start_date).total_seconds())+1):
            yield start_date + timedelta(seconds=n)
    else:
        raise ValueError("Invalid resolution")


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


def Destaggering(date_in, date_fin, path_to_mod_output_arr, path_to_destag_output_folder_arr, name_exp, time_res, path_to_mask_arr):

    start_date = date(int(date_in[0:4]), int(date_in[4:6]), int(date_in[6:8]))
    end_date = date(int(date_fin[0:4]), int(date_fin[4:6]), int(date_fin[6:8]))

    for i, (path_to_mod_output, path_to_destag_output_folder, path_to_mask) in enumerate(zip(path_to_mod_output_arr, path_to_destag_output_folder_arr, path_to_mask_arr)):

        os.makedirs(path_to_destag_output_folder, exist_ok=True)

        listOfFiles = list()
        for (dirpath, dirnames, filenames) in os.walk(path_to_mod_output, followlinks=True):
            print("dirpath: ", dirpath)
            listOfFiles += [os.path.join(dirpath, file) for file in filenames]

        mesh_mask_ds = Dataset(path_to_mask)
        mesh_mask = xr.open_dataset(xr.backends.NetCDF4DataStore(mesh_mask_ds))

        u_mask = mesh_mask.umask.values
        v_mask = mesh_mask.vmask.values
        t_mask = mesh_mask.tmask.values
        u_mask = np.squeeze(u_mask[0, :, :, :])
        v_mask = np.squeeze(v_mask[0, :, :, :])
        t_mask = np.squeeze(t_mask[0, :, :, :])

        for single_date in daterange(start_date, end_date, time_res[i][-1]):
            print(single_date)
            timetag = single_date.strftime("%Y%m%d")
            counter = 0
            print("name_exp: ", name_exp[i])

            u_filename = name_exp[i] + "_" + \
                time_res[i] + "_" + timetag + "_grid_U.nc"
            v_filename = name_exp[i] + "_" + \
                time_res[i] + "_" + timetag + "_grid_V.nc"

            if any(u_filename in s for s in listOfFiles) and any(v_filename in r for r in listOfFiles):
                matching_u = [
                    u_match for u_match in listOfFiles if u_filename in u_match]
                matching_v = [
                    v_match for v_match in listOfFiles if v_filename in v_match]

                U_current = xr.open_dataset(
                    listOfFiles[listOfFiles.index(matching_u[0])])
                V_current = xr.open_dataset(
                    listOfFiles[listOfFiles.index(matching_v[0])])
            else:
                continue

            if time_res[i] == "1d":
                [dim_t, dim_depthu, dim_lat, dim_lon] = U_current.vozocrtx.shape

                u_int = U_current.vozocrtx.values
                u = u_int[0, :, :, :]

                v_int = V_current.vomecrty.values
                v = v_int[0, :, :, :]

                sum_u_mask = u_mask[:, :, 1:]+u_mask[:, :, :(dim_lon-1)]
                sum_u = u[:, :, 1:]+u[:, :, :(dim_lon-1)]
                denominator_u_mask = np.maximum(sum_u_mask, 1)
                destaggered_u = np.zeros(u.shape)
                destaggered_u[:, :, 1:] = sum_u / denominator_u_mask
                destaggered_u = destaggered_u * t_mask

                # destaggering of v
                sum_v_mask = v_mask[:, 1:, :]+v_mask[:, :(dim_lat-1), :]
                sum_v = v[:, 1:, :]+v[:, :(dim_lat-1), :]
                denominator_v_mask = np.maximum(sum_v_mask, 1)
                destaggered_v = np.zeros(v.shape)
                destaggered_v[:, 1:, :] = sum_v / denominator_v_mask
                destaggered_v = destaggered_v*t_mask

                destaggered_U_current = U_current
                if 'nav_lat' in list(destaggered_U_current.keys()):
                    destaggered_U_current = destaggered_U_current.drop(
                        ("nav_lat"))
                if 'nav_lon' in list(destaggered_U_current.keys()):
                    destaggered_U_current = destaggered_U_current.drop(
                        ("nav_lon"))

                destaggered_U_current = destaggered_U_current.assign(
                    destaggered_u=(('depthu', 'y', 'x'), destaggered_u))
                destaggered_U_current = destaggered_U_current.assign(
                    nav_lon=(('y', 'x'), mesh_mask.glamt.values[0, :, :]))
                destaggered_U_current = destaggered_U_current.assign(
                    nav_lat=(('y', 'x'), mesh_mask.gphit.values[0, :, :]))
                destaggered_U_current.destaggered_u.attrs = U_current.vozocrtx.attrs

                destaggered_U_current.to_netcdf(
                    path_to_destag_output_folder + name_exp[i] + "_" + time_res[i] + "_" + timetag + "_grid_U2T.nc")

                # save destaggered v in nc file
                destaggered_V_current = V_current
                if 'nav_lat' in list(destaggered_V_current.keys()):
                    destaggered_V_current = destaggered_V_current.drop(
                        ("nav_lat"))
                if 'nav_lon' in list(destaggered_V_current.keys()):
                    destaggered_U_current = destaggered_V_current.drop(
                        ("nav_lon"))

                destaggered_V_current = destaggered_V_current.assign(
                    destaggered_v=(('depthv', 'y', 'x'), destaggered_v))
                destaggered_V_current = destaggered_V_current.assign(
                    nav_lon=(('y', 'x'), mesh_mask.glamt.values[0, :, :]))
                destaggered_V_current = destaggered_V_current.assign(
                    nav_lat=(('y', 'x'), mesh_mask.gphit.values[0, :, :]))

                destaggered_V_current.destaggered_v.attrs = V_current.vomecrty.attrs
                print(path_to_destag_output_folder +
                      name_exp[i] + "_" + time_res[i] + "_" + timetag + "_grid_V2T.nc")
                destaggered_V_current.to_netcdf(
                    path_to_destag_output_folder + name_exp[i] + "_" + time_res[i] + "_" + timetag + "_grid_V2T.nc")

            elif time_res == "1h":
                [dim_t, dim_lat, dim_lon] = U_current.ssu.shape

                u_int = U_current.ssu.values
                u = u_int[0, :, :]

                v_int = V_current.ssv.values
                v = v_int[0, :, :]

                sum_u_mask = u_mask[:, 1:]+u_mask[:, :(dim_lon-1)]
                sum_u = u[:, 1:]+u[:, :(dim_lon-1)]
                denominator_u_mask = np.maximum(sum_u_mask, 1)
                destaggered_u = np.zeros(u.shape)
                destaggered_u[:, 1:] = sum_u / denominator_u_mask
                destaggered_u = destaggered_u * t_mask

                # destaggering of v
                sum_v_mask = v_mask[1:, :]+v_mask[:(dim_lat-1), :]
                sum_v = v[1:, :]+v[:(dim_lat-1), :]
                denominator_v_mask = np.maximum(sum_v_mask, 1)
                destaggered_v = np.zeros(v.shape)
                destaggered_v[1:, :] = sum_v / denominator_v_mask
                destaggered_v = destaggered_v*t_mask

                destaggered_U_current = U_current
                if 'nav_lat' in list(destaggered_U_current.keys()):
                    destaggered_U_current = destaggered_U_current.drop(
                        ("nav_lat"))
                if 'nav_lon' in list(destaggered_U_current.keys()):
                    destaggered_U_current = destaggered_U_current.drop(
                        ("nav_lon"))

                destaggered_U_current = destaggered_U_current.assign(
                    destaggered_u=(('y', 'x'), destaggered_u))
                destaggered_U_current = destaggered_U_current.assign(
                    nav_lon=(('y', 'x'), mesh_mask.glamt.values[0, :, :]))
                destaggered_U_current = destaggered_U_current.assign(
                    nav_lat=(('y', 'x'), mesh_mask.gphit.values[0, :, :]))
                destaggered_U_current.destaggered_u.attrs = U_current.ssu.attrs

                destaggered_U_current.to_netcdf(
                    path_to_destag_output_folder + name_exp[i] + "_" + time_res[i] + "_" + timetag + "_grid_U2T.nc")

                # save destaggered v in nc file
                destaggered_V_current = V_current
                if 'nav_lat' in list(destaggered_V_current.keys()):
                    destaggered_V_current = destaggered_V_current.drop(
                        ("nav_lat"))
                if 'nav_lon' in list(destaggered_V_current.keys()):
                    destaggered_U_current = destaggered_V_current.drop(
                        ("nav_lon"))

                destaggered_V_current = destaggered_V_current.assign(
                    destaggered_v=(('y', 'x'), destaggered_v))
                destaggered_V_current = destaggered_V_current.assign(
                    nav_lon=(('y', 'x'), mesh_mask.glamt.values[0, :, :]))
                destaggered_V_current = destaggered_V_current.assign(
                    nav_lat=(('y', 'x'), mesh_mask.gphit.values[0, :, :]))

                destaggered_V_current.destaggered_v.attrs = V_current.ssv.attrs
                destaggered_V_current.to_netcdf(
                    path_to_destag_output_folder + name_exp[i] + "_" + time_res[i] + "_" + timetag + "_grid_V2T.nc")


def Find_Nearest_Mod_Point(obs_file, ini_date, name_exp, time_res, path_to_destag_output_folder, depth_obs, last_lat, last_lon, t_mask, mod_grid):

    nearest_point = {}
    for name, obs in obs_file.items():

        print(name)
        timetag = ini_date.strftime("%Y%m%d")

        destaggered_u_file = name_exp + "_" + time_res + "_" + timetag + "_grid_U2T.nc"
        U_current = xr.open_dataset(
            path_to_destag_output_folder + destaggered_u_file)

        vettore_lat = U_current.nav_lat.data[:, 0]
        vettore_lon = U_current.nav_lon.data[0, :]
        vettore_depth = U_current.depthu.data

        depth_osservazione = float(obs[depth_obs])
        array_depth = np.column_stack(U_current.depthu.values)

        nemo_tree = cKDTree(array_depth.T)
        dist_depth, idx_near_obs_depth = nemo_tree.query(
            np.array([depth_osservazione]))

        masked_nav_lat = U_current.nav_lat.values * \
            np.squeeze(t_mask[idx_near_obs_depth, :, :])
        masked_nav_lon = U_current.nav_lon.values * \
            np.squeeze(t_mask[idx_near_obs_depth, :, :])
        masked_nav_lat[masked_nav_lat == 0] = np.nan
        masked_nav_lon[masked_nav_lon == 0] = np.nan

        nemo_lonlat = np.column_stack(
            (masked_nav_lat.ravel(), masked_nav_lon.ravel()))
        nemo_lonlat = nemo_lonlat[~np.isnan(nemo_lonlat).any(axis=1)]
        nemo_tree = cKDTree(nemo_lonlat)
        dist, idx_near_obs = nemo_tree.query(np.column_stack(
            (float(obs[last_lat]), float(obs[last_lon]))))

        if mod_grid == "regular":
            i = np.where(vettore_lon == nemo_lonlat[idx_near_obs][0][1])
            j = np.where(vettore_lat == nemo_lonlat[idx_near_obs][0][0])
        elif mod_grid == "irregular":
            i = np.where(U_current.nav_lon.values ==
                         nemo_lonlat[idx_near_obs][0][1])[1]
            j = np.where(U_current.nav_lat.values ==
                         nemo_lonlat[idx_near_obs][0][0])[0]
        else:
            sys.exit("mod_grid must be set regular or irregular")
        k = np.where(vettore_depth ==
                     U_current.depthu.values[idx_near_obs_depth])

        nearest_point[name] = {'lon_idx': i, 'lat_idx': j, 'depth_idx': k}
        #print("nearest point: ",nearest_point[name])
        U_current.close()

    return nearest_point


def Save_Mod_Ts(obs_file, start_date, end_date, name_exp, date_in, date_fin, temp_res, path_to_destag_output_folder, nearest_point, path_to_output_mod_ts_folder, mod_grid, time_res_to_average, name_stat, CMEMS_code, WMO_code, path_to_obs_file):
    mod_file = {}
    mod_dir_file = {}
    mod_time_file = {}

    mod_dict = {}
    averaged_mod_dict = {}
    for name, obs in obs_file.items():
        print(name)
        for single_date in daterange(start_date, end_date, temp_res[-1]):
            print(single_date.strftime("%Y-%m-%d"))
            timetag = single_date.strftime("%Y%m%d")

            destaggered_u_file = name_exp + "_" + temp_res + "_" + timetag + "_grid_U2T.nc"
            destaggered_v_file = name_exp + "_" + temp_res + "_" + timetag + "_grid_V2T.nc"
            print(path_to_destag_output_folder + destaggered_u_file)
            try:
                U_current = xr.open_dataset(
                    path_to_destag_output_folder + destaggered_u_file)
                V_current = xr.open_dataset(
                    path_to_destag_output_folder + destaggered_v_file)
            except Exception:
                append_value(mod_file, name, np.nan)
                append_value(mod_dir_file, name, np.nan)
                append_value(mod_time_file, name, single_date)
                continue

            if mod_grid == "regular":
                i = nearest_point[name]['lon_idx'][0][0]
                j = nearest_point[name]['lat_idx'][0][0]
            elif mod_grid == "irregular":
                i = nearest_point[name]['lon_idx'][0]
                j = nearest_point[name]['lat_idx'][0]
            else:
                sys.exit("mod_grid must be set regular or irregular")

            k = nearest_point[name]['depth_idx'][0][0]
            u = U_current.destaggered_u.values[k, j, i]
            U_current.close()
            v = V_current.destaggered_v.values[k, j, i]
            V_current.close()
            velocity = math.sqrt(u**2 + v**2)
            print("velocity: ", velocity)
            mod_direction = wind_direction(u, v)
            append_value(mod_file, name, velocity)
            append_value(mod_dir_file, name, mod_direction)
            append_value(mod_time_file, name, single_date)

        array_name = np.array([obs[name_stat], obs[CMEMS_code], obs[WMO_code]])
        print(array_name)
        boolArr = np.where(array_name != "_")
        output_mod_file = array_name[boolArr][0] + "_" + \
            date_in + "_" + date_fin + "_" + temp_res + "_mod.nc"

        dataset_obs = xr.open_dataset(obs[path_to_obs_file])
        if temp_res == "1d":
            averaged_time = dataset_obs['TIME'].resample(
                TIME="1D").mean(skipna=True)
        if temp_res == "1h":
            averaged_time = dataset_obs['TIME'].resample(
                TIME="1H").mean(skipna=True)
        averaged_time = averaged_time.sel(TIME=slice(start_date, end_date))

        # set the datetime column as the index of the DataFrame
        print("mod file name: ", mod_file[name])
        mod_ds = pd.DataFrame(
            {'datetime': averaged_time['TIME'].data[:], 'velocity': mod_file[name], 'direction': mod_dir_file[name]})
        mod_ds.set_index('datetime', inplace=True)

        if len(mod_time_file[name][:]) > 0:
            if not isinstance(mod_ds.index, pd.DatetimeIndex):
                mod_ds.index = pd.to_datetime(mod_ds.index)
        else:
            continue

        # resample the DataFrame by day, calculating the mean of each group
        averaged_mod_ds = mod_ds.resample(time_res_to_average).mean()
        averaged_mod_velocities = averaged_mod_ds['velocity'].tolist()
        print("averaged mod velocities: ", averaged_mod_velocities)
        averaged_mod_directions = averaged_mod_ds['direction'].tolist()
        averaged_mod_times = averaged_mod_ds.index.tolist()
        if not np.isnan(averaged_mod_velocities).all():
            print("sono qui")
            mod_dict[array_name[boolArr][0]] = {
                'datetime': mod_time_file[name], 'velocity': mod_file[name], 'direction': mod_dir_file[name]}
            averaged_mod_dict[array_name[boolArr][0]] = {
                'datetime': averaged_mod_times, 'velocity': averaged_mod_velocities, 'direction': averaged_mod_directions}

        averaged_mod_ds = xr.Dataset(data_vars=dict(TIME=(['TIME'], averaged_mod_dict[array_name[boolArr][0]]['datetime'][:]), mod_vel=(
            ['TIME'], averaged_mod_dict[array_name[boolArr][0]]['velocity'][:]), mod_direction=(['TIME'], averaged_mod_dict[array_name[boolArr][0]]['direction'][:])))
        # averaged_mod_ds=mod_ds.resample(TIME=time_res_to_average).mean(skipna=True)
        averaged_mod_ds.to_netcdf(
            path=path_to_output_mod_ts_folder + output_mod_file)

    return mod_dict, averaged_mod_dict
