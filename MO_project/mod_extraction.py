import warnings
warnings.filterwarnings("ignore")
import netCDF4 as nc4
import os.path
import numpy as np
import xarray as xr
from datetime import date
import matplotlib.pyplot as plt
import csv
import argparse

from core import Find_Nearest_Mod_Point, Save_Mod_Ts


def parse_args():
    parser = argparse.ArgumentParser(
        description='Extraction of Mod Time Series')
    parser.add_argument('mod_date_in', type=str,
                        help='start date from which to extract')
    parser.add_argument('mod_date_fin', type=str,
                        help='end date until which to extract')
    parser.add_argument('path_to_metadata_obs_file', type=str,
                        help='Path to the metadata file of accepted observations')
    parser.add_argument('time_res_model', type=str,
                        help='Time resolution of destaggered model data')
    parser.add_argument('time_res_model_to_average', type=str,
                        help='Time resolution to average the destaggered model data')
    parser.add_argument('mod_exp_names', type=str,
                        help='Names of the model experiments')
    parser.add_argument('depth_obs', type=float,
                        help='Depth at (or around) which model data are extracted')
    parser.add_argument('path_to_mask', type=str,
                        help='Path to mask file of the model experiments')
    parser.add_argument('mod_grid', type=str,
                        help='Set if grid is regular or irregular')
    parser.add_argument('path_to_input_model_folder', type=str,
                        help='Path to the input destaggered model folder')
    parser.add_argument('path_to_out_mod_ts', type=str,
                        help='Path to the output extracted model folder')
    parser.add_argument('path_to_plot_mod_ts', type=str,
                        help='Path to the output model plot folder')
    args = parser.parse_args()
    return args

def main(args):
    date_in = args.mod_date_in
    date_fin = args.mod_date_fin
    path_to_metadata_obs_file = args.path_to_metadata_obs_file
    time_res_model_arr = args.time_res_model.split()
    time_res_model_to_average_arr = args.time_res_model_to_average.split()
    name_exp_arr = args.mod_exp_names.split()
    depth_obs_set = args.depth_obs
    path_to_mask_arr = args.path_to_mask.split()
    mod_grid_arr = args.mod_grid.split()
    path_to_input_mod_folder_arr = args.path_to_input_model_folder.split()
    path_to_out_mod_ts_arr = args.path_to_out_mod_ts.split()
    path_to_plot_mod_ts_arr = args.path_to_plot_mod_ts.split()
    
    start_date = date(int(date_in[0:4]),int(date_in[4:6]) , int(date_in[6:8]))
    end_date = date(int(date_fin[0:4]),int(date_fin[4:6]) , int(date_fin[6:8]))
    
    obs_file = {}
    
    with open(path_to_metadata_obs_file) as f:
        header_row = f.readline().rstrip()

        # remove the semicolon at the end of the header row, if present
        if header_row[-1] == ';':
            header_row = header_row[:-1]
        #first_line = f.readline().rstrip().strip(";").strip()
        #column_names = first_line.replace(" ","")
        column_names = header_row.split(';')
        column_names.append('depth')
        #reader = csv.reader(filter(lambda row: row[0]!='#',f), delimiter=';')
        reader = csv.reader(filter(lambda row: row and row[-1] != '', (line.rstrip().strip(';') for line in f if not line.startswith('#'))), delimiter=';')
        for count, row in enumerate(reader):
            if not (row):    
                continue
            row_1=[val.rstrip() for val in row]
            print(row)
            print(row_1)
            print(depth_obs_set)
            row_1.append(depth_obs_set)
            print(row)
            print("columns names: ", column_names)
            obs_file[count]= dict(zip(column_names, row_1))
            print(obs_file[count])

        last_lat = column_names[0]
        last_lon = column_names[1]
        num_vlevs = column_names[2]
        num_sfc_levs = column_names[3]
        name_stat = column_names[4]
        CMEMS_code = column_names[5]
        WMO_code = column_names[6]
        time_period = column_names[7]
        fields_list = column_names[8]
        qf_value = column_names[9]
        path_to_obs_file = column_names[10]
        depth_obs = column_names[11]
    
    for folder_nc, folder_plot in zip(path_to_out_mod_ts_arr,path_to_plot_mod_ts_arr):
        os.makedirs(folder_nc, exist_ok=True)
        os.makedirs(folder_plot, exist_ok=True)
        
    for i, (mask_file, input_mod_folder) in enumerate(zip(path_to_mask_arr, path_to_input_mod_folder_arr)):

        mesh_mask = xr.open_dataset(mask_file)
        t_mask = mesh_mask.tmask.values
        t_mask = np.squeeze(t_mask[0, :, :, :])

        nearest_point = {}
        nearest_point = Find_Nearest_Mod_Point(obs_file, start_date, name_exp_arr[i], time_res_model_arr[i], input_mod_folder, depth_obs, last_lat, last_lon, t_mask, mod_grid_arr[i])
        mod_dict, averaged_mod_dict = Save_Mod_Ts(obs_file, start_date, end_date, name_exp_arr[i], date_in, date_fin, time_res_model_arr[i], input_mod_folder, nearest_point, path_to_out_mod_ts_arr[i], mod_grid_arr[i], time_res_model_to_average_arr[i],name_stat, CMEMS_code, WMO_code, path_to_obs_file)

        for dict_key, dict_value in averaged_mod_dict.items():
            fig, (ax1, ax2) = plt.subplots(2)
            fig.subplots_adjust(hspace=1)
            fig.suptitle('Mooring: {} (model experiment {})'.format(dict_key, name_exp_arr[i]))
            ax1.plot(mod_dict[dict_key]['datetime'][:], mod_dict[dict_key]['velocity'][:])
            ax1.tick_params(axis='x', rotation=45, labelsize=7) 
            ax2.plot(dict_value['datetime'][:],dict_value['velocity'][:])
            ax2.tick_params(axis='x', rotation=45, labelsize=7)
            filename = dict_key + '_plot_mod_ts.png'
            fig.savefig(path_to_plot_mod_ts_arr[i] + filename)

            fig_dir, (ax1_dir, ax2_dir) = plt.subplots(2)
            fig_dir.subplots_adjust(hspace=1)
            fig_dir.suptitle('Mooring: {} (model experiment {})'.format(dict_key, name_exp_arr[i]))
            ax1_dir.plot(mod_dict[dict_key]['datetime'][:], mod_dict[dict_key]['direction'][:])
            ax1_dir.tick_params(axis='x', rotation=45, labelsize=7) 
            ax2_dir.plot(dict_value['datetime'][:],dict_value['direction'][:])
            ax2_dir.tick_params(axis='x', rotation=45, labelsize=7)
            filename = dict_key + '_plot_dir_ts.png'
            fig_dir.savefig(path_to_plot_mod_ts_arr[i] + filename)
            
if __name__ == "__main__":

    args = parse_args()
    main(args)