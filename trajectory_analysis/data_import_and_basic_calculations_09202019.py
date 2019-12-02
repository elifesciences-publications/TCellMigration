import trajectory_analysis_functions_09192019
import numpy

def import_data_and_measure( trajectory_file ):
	
	master_trajectory_dict = trajectory_analysis_functions_09192019.import_data_merged_file(trajectory_file)

	for experiment in master_trajectory_dict:
		
		if 'Gautreau' in experiment:
			for treatment in master_trajectory_dict[experiment]:
				for sample in master_trajectory_dict[experiment][treatment]:
					subsampled_trajs = trajectory_analysis_functions_09192019.subsample_timepoints(master_trajectory_dict[experiment][treatment][sample], subsampling_factor=4)
					master_trajectory_dict[experiment][treatment][sample] = subsampled_trajs
		if 'fish' in experiment:
			for treatment in master_trajectory_dict[experiment]:
				
						
				if 'highfreq' in treatment:
					for sample in master_trajectory_dict[experiment][treatment]:
						
						subsampled_trajs = trajectory_analysis_functions_09192019.subsample_timepoints(master_trajectory_dict[experiment][treatment][sample], subsampling_factor=4)
						master_trajectory_dict[experiment][treatment][sample] = subsampled_trajs
	
	trajectory_dict_polar = trajectory_analysis_functions_09192019.trajectories_polar( master_trajectory_dict )

	corr_times_dict, speeds_dict = trajectory_analysis_functions_09192019.correlation_times_lengths_by_cell( trajectory_dict_polar, calculation='times', overlap=True )
	corr_lengths_dict, speeds_dict2 = trajectory_analysis_functions_09192019.correlation_times_lengths_by_cell( trajectory_dict_polar, calculation='lengths', overlap=False )
	step_angle_coupling_dict = trajectory_analysis_functions_09192019.length_angle_couplings_by_cell( trajectory_dict_polar )
	trajectory_dict_polar_interp = trajectory_analysis_functions_09192019.interpolate_timepoints( trajectory_dict_polar )

	return master_trajectory_dict, trajectory_dict_polar, trajectory_dict_polar_interp, speeds_dict, corr_times_dict, corr_lengths_dict, step_angle_coupling_dict

def import_data_and_measure_2x_subsampled( trajectory_file ):
	
	master_trajectory_dict = trajectory_analysis_functions_09192019.import_data_merged_file(trajectory_file)

	for experiment in master_trajectory_dict:
		if 'Gerard' in experiment:
			for treatment in master_trajectory_dict[experiment]:
				for sample in master_trajectory_dict[experiment][treatment]:
					subsampled_trajs = trajectory_analysis_functions_09192019.subsample_timepoints(master_trajectory_dict[experiment][treatment][sample], subsampling_factor=2)
					master_trajectory_dict[experiment][treatment][sample] = subsampled_trajs
		if 'Gautreau' in experiment:
			for treatment in master_trajectory_dict[experiment]:
				for sample in master_trajectory_dict[experiment][treatment]:
					subsampled_trajs = trajectory_analysis_functions_09192019.subsample_timepoints(master_trajectory_dict[experiment][treatment][sample], subsampling_factor=6)
					master_trajectory_dict[experiment][treatment][sample] = subsampled_trajs
		if 'fish' in experiment:
			for treatment in master_trajectory_dict[experiment]:
				if 'highfreq' not in treatment:
					for sample in master_trajectory_dict[experiment][treatment]:
						subsampled_trajs = trajectory_analysis_functions_09192019.subsample_timepoints(master_trajectory_dict[experiment][treatment][sample], subsampling_factor=2)
						master_trajectory_dict[experiment][treatment][sample] = subsampled_trajs
						
				else:
					for sample in master_trajectory_dict[experiment][treatment]:
						
						subsampled_trajs = trajectory_analysis_functions_09192019.subsample_timepoints(master_trajectory_dict[experiment][treatment][sample], subsampling_factor=8)
						master_trajectory_dict[experiment][treatment][sample] = subsampled_trajs
	
	trajectory_dict_polar = trajectory_analysis_functions_09192019.trajectories_polar( master_trajectory_dict )

	corr_times_dict, speeds_dict = trajectory_analysis_functions_09192019.correlation_times_lengths_by_cell( trajectory_dict_polar, calculation='times', overlap=True )
	corr_lengths_dict, speeds_dict = trajectory_analysis_functions_09192019.correlation_times_lengths_by_cell( trajectory_dict_polar, calculation='lengths', overlap=False )
	step_angle_coupling_dict = trajectory_analysis_functions_09192019.length_angle_couplings_by_cell( trajectory_dict_polar )
	trajectory_dict_polar_interp = trajectory_analysis_functions_09192019.interpolate_timepoints( trajectory_dict_polar )

	return master_trajectory_dict, trajectory_dict_polar, trajectory_dict_polar_interp, speeds_dict, corr_times_dict, corr_lengths_dict, step_angle_coupling_dict