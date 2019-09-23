include: "Snakefile_accessory_v4.py"
from glob import glob
import os



rule all:
  	input: OUTPUT_BASE + '_MaxZ_488nm.mp4'#, OUTPUT_BASE + '_MaxZ_488nm_bf_overlay.mp4', OUTPUT_BASE + '_MaxZ_488nm_bf_overlay_bgsub.mp4'#, OUTPUT_BASE + '_t0.tif'
	params: name='all', mem='1024'

rule stitch:
	input: files_in_trange2
	output: OUTPUT_BASE + '_{t}_raw.hdf5'
	params: name='stitch',mem='10000'
	run:
		wdir = os.path.dirname(str(output))
		trange = get_trange
		shell("mkdir -p {wdir} && "
			  "python stitch_tiles4.py {RAW_IMAGE_FOLDER} {METADATA_FILE} {wdir} {wildcards.t} {MAX_TSTEPS}")

rule t0_stack:
	input: OUTPUT_BASE + '_block0_raw.hdf5'
	output: OUTPUT_BASE + '_t0.tif'
	params: name='save_t0',mem='10000'
	run:
		wdir = os.path.dirname(str(output))
		shell("python save_t0_tif.py {input} {output}")
		
rule z_project:
	input: rules.stitch.output
	output: OUTPUT_BASE + '_{t}_MaxZ.hdf5', OUTPUT_BASE + '_{t}_MaxZ_slice0.hdf5'
	params: name='MaxZ',mem='10000'
	run:
		shell("python max_z_project2.py {input}")

rule combine_timesteps_photobleach_adjust:
	input: expand(OUTPUT_BASE +'_{t}_MaxZ.hdf5', t=TBLOCK_LIST)
	output: OUTPUT_BASE + '_MaxZ.tif', OUTPUT_BASE + '_MaxZ_photobleachadj.tif'
	params: name='combineMaxZ',mem='10000'
	run:
		shell("python combine_time_blocks_photobleach_adj.py {input} {UPPER_THRESH} {MAX_TSTEPS}")
		
rule drift_correction:
	input: OUTPUT_BASE + '_MaxZ_photobleachadj.tif'
	output: OUTPUT_BASE + '_MaxZ_driftcorr.tif', OUTPUT_BASE + '_drift_trajectory.txt'
	params: name='drift_corr',mem='10000'
	run:
		shell("python drift_subtraction2.py {input[0]} {output[0]} {output[1]} 25 {DRIFT_COORDS}")

#rule cross_corr_drift_correction:
#	input: OUTPUT_BASE + '_{t}_MaxZ.hdf5', OUTPUT_BASE + '_block0_MaxZ_slice0.hdf5'
#	output: OUTPUT_BASE + '_{t}_MaxZ_driftcorr.hdf5', OUTPUT_BASE + '_{t}_drift_trajectory.txt'
#	params: name='drift_corr',mem='10000'
#	run:
#		shell("python cross_correlation_drift_subtraction.py {input[0]} {output[0]} {input[1]} {output[1]} 99")



rule background_subtraction_segmentation:
	input: OUTPUT_BASE + '_MaxZ_driftcorr.tif', OUTPUT_BASE + '_drift_trajectory.txt'
	output: OUTPUT_BASE + '_MaxZ_bgsub.tif', OUTPUT_BASE + '_MaxZ_driftcorr_frame.tif', OUTPUT_BASE + '_MaxZ_bgsub_binary.tif', OUTPUT_BASE + '_MaxZ_bgsub_labels.tif',OUTPUT_BASE + '_parameters.txt',OUTPUT_BASE + '_centroids.txt'
	params: name='bgsub_segment',mem='50000'
	run:
		shell("python background_subtract.py {input[0]} {input[1]} {output[0]} {output[1]} {output[2]} {output[3]} {output[4]} {output[5]} 0 {UPPER_THRESH} 95")

rule stitch_brightfield:
	input: get_bf_input
	output: OUTPUT_BASE + '_bf_stitched.tif'
	params: name='stitchBF',mem='50000'
	run:
		shell("python stitch_bf_frame4.py {BRIGHTFIELD_FOLDER} {METADATA_FILE_BF} {output} {input[0]} {METADATA_FILE}")
	
rule onechannel_movie:
	input: OUTPUT_BASE + '_MaxZ_bgsub.tif'
	output: OUTPUT_BASE + '_MaxZ_488nm.mp4'
	params: name='movie_488',mem='50000'
	run:
		shell("python make_onechannel_movie.py {input} {output} 0")

rule overlay_movie:
	input: OUTPUT_BASE + '_bf_stitched.tif', OUTPUT_BASE + '_MaxZ_bgsub.tif'
	output: OUTPUT_BASE + '_MaxZ_488nm_bf_overlay_bgsub.mp4'
	params: name='movie_overlay',mem='50000'
	run:
		shell("python make_overlay_movie2.py {input[0]} {input[1]} {output} 0 {UPPER_THRESH} 500")

rule overlay_movie2:
	input: OUTPUT_BASE + '_bf_stitched.tif', OUTPUT_BASE + '_MaxZ_driftcorr_frame.tif'
	output: OUTPUT_BASE + '_MaxZ_488nm_bf_overlay.mp4'
	params: name='movie_overlay',mem='50000'
	run:
		shell("python make_overlay_movie2.py {input[0]} {input[1]} {output} 20 {UPPER_THRESH} 500")