import argparse
import configparser
import datetime

class Configuration:
    def __init__(self, config_file):
        config = configparser.ConfigParser()
        config.read(config_file)
        
        self.paths = config['link_section']['path_to_mod_files'].split(',')
        self.old_names = config['link_section']['old_name_exp'].split(',')
        self.new_names = config['link_section']['new_name_exp'].split(',')
        self.link_date_in = datetime.datetime.strptime(
            config['link_section']['date_in'], '%Y%m%d').date()
        self.link_date_fin = datetime.datetime.strptime(
            config['link_section']['date_fin'], '%Y%m%d').date()
        self.time_res = config['link_section']['time_res_model'].split(',')
        self.out_paths = config['link_section']['path_to_out_model_folder'].split(',')
        
        self.destag_date_in = config['destaggering_section']['date_in']
        self.destag_date_fin = config['destaggering_section']['date_fin']
        self.input_paths = config['destaggering_section']['path_to_input_model_folder'].split(',')
        self.mask_paths = config['destaggering_section']['path_to_mask'].split(',')
        self.destag_time_res = config['destaggering_section']['time_res_input_model'].split(',')
        self.exp_names = config['destaggering_section']['name_exp'].split(',')
        self.path_to_out_destag_model_folder = config['destaggering_section']['path_to_out_destag_model_folder'].split(',')
        
        self.obs_date_in = config['obs_extraction_section']['date_in']
        self.obs_date_fin = config['obs_extraction_section']['date_fin']
        self.path_to_metadata_obs_file = config['obs_extraction_section']['path_to_metadata_obs_file']
        self.time_res_to_average = config['obs_extraction_section']['time_res_to_average']
        self.depth_obs = config['obs_extraction_section']['depth_obs']
        self.nan_treshold = config['obs_extraction_section']['nan_treshold']
        self.path_to_accepted_metadata_obs_file = config['obs_extraction_section']['path_to_accepted_metadata_obs_file']
        self.path_to_out_obs_ts = config['obs_extraction_section']['path_to_out_obs_ts']
        self.path_to_plot_ts = config['obs_extraction_section']['path_to_plot_ts']
        
        self.mod_date_in = config['mod_extraction_section']['date_in']
        self.mod_date_fin = config['mod_extraction_section']['date_fin']
        self.path_to_input_metadata_obs_file = config['mod_extraction_section']['path_to_input_metadata_obs_file']
        self.time_res_model = config['mod_extraction_section']['time_res_model'].split(',')
        self.time_res_model_to_average = config['mod_extraction_section']['time_res_model_to_average'].split(',')
        self.mod_exp_names = config['mod_extraction_section']['name_exp'].split(',')
        self.depth_obs_for_mod = config['mod_extraction_section']['depth_obs']
        self.path_to_mask = config['mod_extraction_section']['path_to_mask'].split(',')
        self.mod_grid = config['mod_extraction_section']['grid'].split(',')
        self.path_to_input_model_folder = config['mod_extraction_section']['path_to_input_model_folder'].split(',')
        self.path_to_out_mod_ts = config['mod_extraction_section']['path_to_out_mod_ts'].split(',')
        self.path_to_plot_mod_ts = config['mod_extraction_section']['path_to_plot_mod_ts'].split(',')
        
        self.plot_date_in = config['plot_statistics_section']['date_in']
        self.plot_date_fin = config['plot_statistics_section']['date_fin']
        self.time_res_plot = config['plot_statistics_section']['time_res'].split(',')
        self.path_to_in_mod_ts = config['plot_statistics_section']['path_to_in_mod_ts'].split(',')
        self.path_to_in_obs_ts = config['plot_statistics_section']['path_to_in_obs_ts']
        self.label_plot = config['plot_statistics_section']['label_plot'].split(',')
        self.time_res_axis = config['plot_statistics_section']['time_res_axis']
        self.path_to_output_exp = config['plot_statistics_section']['path_to_exp'].split(',')
        self.path_to_comparison = config['plot_statistics_section']['path_to_comparison']
        self.path_to_accepted_metadata_obs_file_for_plot = config['plot_statistics_section']['path_to_accepted_metadata_obs_file']
        
        
def parse():
    # Create ArgumentParser object
    parser = argparse.ArgumentParser()

    # Add --link command
    parser.add_argument('--link', action='store_true', help='Link Model Files')
    
    # Add --destaggering command
    parser.add_argument('--destaggering', action='store_true', help='Destag Model Files')   
    
    # Add --obs_extract command
    parser.add_argument('--obs_extract', action='store_true', help='Extract Obs Time Series')
    
    # Add --mod_extract command
    parser.add_argument('--mod_extract', action='store_true', help='Extract Mod Time Series')
    
    # Add --plot_stats command
    parser.add_argument('--plot_stats', action='store_true', help='Plot Statistics')

    # Parse arguments
    args = parser.parse_args()
    return args