from full_fov_video_no_trajectories import make_cell_movie

base_folder = '/mnt/f/SPIM/_03252019/lckgfp_dob03122018_fish5_488_trial_2'
timeseries = base_folder + '/lckgfp_dob03122018_fish5_488_trial_2_MaxZ_driftcorr_frame_regtif_tail_right_scalebar.tif'
output_base = base_folder + '/lckgfp_dob03122018_fish5_488_trial_2_movie'

#drift_adjustment(timeseries,timeseries_ds,drift_traj_file,pixel_thresh)
make_cell_movie(timeseries,output_base,timestep=45,ntmax=1000)