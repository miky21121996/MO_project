import subprocess
from .cli import parse, Configuration
from .core import Link_Files, Destaggering


def main():
    args = parse()
    config = Configuration('/work/oda/mg28621/MO_project/MO_project/configuration.ini')
    if args.link:
        Link_Files(config.paths, config.old_names, config.new_names, config.link_date_in,
                config.link_date_fin, config.time_res, config.out_paths)

    if args.destaggering:
        Destaggering(config.destag_date_in, config.destag_date_fin, config.input_paths, config.path_to_out_destag_model_folder, config.exp_names, config.destag_time_res, config.mask_paths)
        
    if args.obs_extract:
        subprocess.run(['bsub', '-K', '-n', '1', '-q', 's_long', '-J', 'CURVAL', '-e', 'aderr_0', '-o', 'adout_0', '-P',\
            '0510', 'python', 'obs_extraction.py', config.obs_date_in, config.obs_date_fin, config.path_to_metadata_obs_file,\
            config.time_res_to_average, config.depth_obs, config.nan_treshold,\
            config.path_to_accepted_metadata_obs_file, config.path_to_out_obs_ts, config.path_to_plot_ts])
        
    if args.mod_extract:
        subprocess.run(['bsub', '-K', '-n', '1', '-q', 's_long', '-J', 'CURVAL', '-e', 'aderr_0', '-o', 'adout_0', '-P',\
            '0510', 'python', 'mod_extraction.py', config.mod_date_in, config.mod_date_fin, config.path_to_input_metadata_obs_file,\
            " ".join(config.time_res_model), " ".join(config.time_res_model_to_average), " ".join(config.mod_exp_names), config.depth_obs_for_mod, " ".join(config.path_to_mask),\
            " ".join(config.mod_grid), " ".join(config.path_to_input_model_folder), " ".join(config.path_to_out_mod_ts), " ".join(config.path_to_plot_mod_ts)])
        
    if args.plot_stats:
        subprocess.run(['bsub', '-K', '-n', '1', '-q', 's_long', '-J', 'CURVAL', '-e', 'aderr_0', '-o', 'adout_0', '-P',\
            '0510', 'python', 'plot_statistics.py', config.plot_date_in, config.plot_date_fin, " ".join(config.time_res_plot), " ".join(config.path_to_in_mod_ts), config.path_to_in_obs_ts, " ".join(config.label_plot),\
            config.time_res_axis, " ".join(config.path_to_output_exp), config.path_to_comparison, config.path_to_input_metadata_obs_file])


if __name__ == '__main__':
    main()
