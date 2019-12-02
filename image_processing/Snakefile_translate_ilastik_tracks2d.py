include: "Snakefile_accessory_v4.py"
from glob import glob
import os
OUTPUT_BASE = OUTPUT_PATH + '/' + SAMPLE_NAME

rule all:
  	input: OUTPUT_BASE + '_manual_tracks.txt'#, OUTPUT_BASE + '_MaxZ_488nm_bf_overlay_manual_tracking.mp4'
	params: name='all', mem='1024'
		
rule track_2D:
	input: OUTPUT_BASE + '_MaxZ_bgsub.tif', TRACKING_MASK_FILE, MERGER_FILE
	output: OUTPUT_BASE + '_{t}_tracks2d.txt'
	params: name='find_centroids',mem='10000'
	run:
		TRANGE = get_trange(wildcards).strip(',').split(',')
		TSTART = TRANGE[0]
		TSTOP = str(int(TRANGE[-1]) + 1)
		print(TSTART,TSTOP,TRANGE)
		shell("python translate_ilastik_tracking_2d.py {input[0]} {input[1]} {input[2]} {output} {TSTART} {TSTOP}")
		

rule concatenate_tracks:
	input: expand(OUTPUT_BASE + '_{t}_tracks2d.txt', t=TBLOCK_LIST)
	output: OUTPUT_BASE + '_manual_tracks.txt'
	params: name='concatenate_tracks'
	run:
		shell("python concatenate_tracks.py {input} {output}")