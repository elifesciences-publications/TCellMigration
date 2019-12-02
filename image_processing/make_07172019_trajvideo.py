from full_fov_video_with_trajectories import make_cell_movie
from drift_subtraction_from_trajectory import drift_adjustment

pixel_thresh = 1200

base_folder = '/mnt/f/SPIM/_07172019/lckgfp_dob07072019_fish3_488_trial3_3'
timeseries = base_folder + '/output/lckgfp_dob07072019_fish3_488_trial3_3_MaxZ_Pos0.tif'
timeseries_ds = base_folder + '/output/lckgfp_dob07072019_fish3_488_trial3_3_MaxZ_Pos0_gain_ds.tif'
trajectory_file = base_folder + '/output/lckgfp_dob07072019_fish3_488_trial3_3_tracks2d_ds.txt'
drift_traj_file = base_folder + '/output/lckgfp_dob07072019_fish3_488_trial3_3_drift_trajectory.txt'
merger_file = base_folder + '/output/mergers.csv'
output_base = base_folder + '/output/lckgfp_dob07072019_fish3_488_trial3_3'

#drift_adjustment(timeseries,timeseries_ds,drift_traj_file,pixel_thresh)
make_cell_movie(timeseries_ds,trajectory_file,merger_file,output_base,timestep=12,ntmax=5*180+1)