import utilities2
import numpy
from glob import glob

###Execution: snakemake --config sample_name=SAMPLE_NAME date=DATE max_tstep=1000 upper_thresh=99.8 data_dir="/mnt/g/SPIM/" drift_coords="No_correction" --cores 12 --snakefile Snakefile_v4.py

SAMPLE_NAME = config["sample_name"]
ACQUISITION_DATE = config["date"]
MAX_TSTEPS = config["max_tstep"] ###Process up until this timestep or the end of acquisition, whichever is smaller
if config["drift_coords"] == "No_correction":
	DRIFT_COORDS = "No_correction"
	DRIFT_FLAG = False
else:
	DRIFT_COORDS = [str(l) for l in list(config["drift_coords"])] ###Pixel coordinates of two pixel spots to track
	DRIFT_FLAG = True
	print(DRIFT_COORDS)
UPPER_THRESH = config["upper_thresh"]
DATA_DIR = config["data_dir"]
#8/1 DSET 1: 219 1458 709 1009
####Note that appropriate execution requires a metadata file in the Pos0 subfolder
#DATA_DIR = '/mnt/g/SPIM/'
#DATA_DIR = '/mnt/e/google_drive/rgb_spim_data/'
RAW_IMAGE_FOLDER =  DATA_DIR + ACQUISITION_DATE + '/' + SAMPLE_NAME
OUTPUT_PATH = DATA_DIR + ACQUISITION_DATE + '/' + SAMPLE_NAME
METADATA_FILE = DATA_DIR + ACQUISITION_DATE + '/' + SAMPLE_NAME + '/Pos0/metadata.txt'
#POSITION_FILE = DATA_DIR + ACQUISITION_DATE + '/' + SAMPLE_NAME + '/' + 'PositionList.pos'
BRIGHTFIELD_FOLDER = DATA_DIR + ACQUISITION_DATE + '/' + SAMPLE_NAME + '_bf'
METADATA_FILE_BF = BRIGHTFIELD_FOLDER + '/Pos0/metadata.txt'


OBJECT_MASK_FILE = OUTPUT_PATH + '/' + SAMPLE_NAME + '_MaxZ_bgsub_Object_Predictions2.h5'
#TRACKING_MASK_FILE = OUTPUT_PATH + '/' + SAMPLE_NAME + '_MaxZ_bgsub_Manual_Tracking.h5'
TRACKING_MASK_FILE = OUTPUT_PATH + '/' + SAMPLE_NAME + '_Manual_Tracking.h5'
#TRACKING_MASK_FILE = OUTPUT_PATH + '/' + SAMPLE_NAME + '_MaxZ_driftcorr_frame_Manual_Tracking.h5'

MERGER_FILE = OUTPUT_PATH + '/mergers.csv'

TBLOCK_LIST = ['block0','block1','block2','block3','block4','block5','block6','block7','block8','block9','block10','block11']

OUTPUT_BASE = OUTPUT_PATH + '/' + SAMPLE_NAME

def get_tranges():
	
	metadata_dict = utilities2.get_stack_metadata( METADATA_FILE )
	
	ntsteps_acquired = float(metadata_dict["Frames"])
	
	ntsteps = min(ntsteps_acquired, int(MAX_TSTEPS))
	
	blocksize = int(numpy.ceil(ntsteps/12))
	
	trange_dict = {}
	
	for block in range(12):
		
		blockname = 'block' + str(block)
		
		trange_dict[blockname] = ''
		
		for num in numpy.arange( blocksize*block, min(blocksize*(block+1),int(MAX_TSTEPS)), 1):
		
			trange_dict[blockname] += str(num) + ','
	
	return trange_dict
	
	
def all_maxz_blocks(wildcards):

	trange_dict = get_tranges()
	
	file_name_list = []
	
	for trange in trange_dict:
		file_name = OUTPUT_PATH + '/' + SAMPLE_NAME + '_' + trange + '_MaxZ_driftcorr.hdf5'
		file_name_list.append(file_name)
	
	return file_name_list

def get_trange( wildcards ):
	
	trange_dict = get_tranges()
	
	trange = trange_dict[wildcards.t]
	
	return trange
	
def files_in_trange( wildcards ):
	
	trange_dict = get_tranges()
	
	trange = trange_dict[wildcards.t].split(',')
	
	pos_list = glob(RAW_IMAGE_FOLDER + '/Pos[0-9]*')
	
	filenames = []
	
	for pos_name in pos_list:
		for t in trange:
			tstr = str(t)
			
			while len(tstr) < 8.5:
			
				tstr = '0' + tstr
				
			filenames.append( pos_name + '/img_' + tstr + '_Default_000.tif' )
	
	
	return filenames


def files_in_trange2( wildcards ):
	
	trange_dict = get_tranges()
	
	trange = trange_dict[wildcards.t].split(',')
	
	pos_list = glob(RAW_IMAGE_FOLDER + '/Pos[0-9]*')
	
	filenames = []
	
	for pos_name in pos_list:
		for t in trange:
			tstr = str(t)
			
			while len(tstr) < 8.5:
			
				tstr = '0' + tstr
			slice_filenames = glob( pos_name 	+ '/img_' + tstr + '_Default_*.tif' )
			for s in slice_filenames:
				filenames.append( s )	
	return filenames

def get_stitch_bf_input( wildcards ):
	
	parameter_file = OUTPUT_PATH + '/' + SAMPLE_NAME + '_parameters.txt'
	bf_folders = glob( BRIGHTFIELD_FOLDER + '/Pos[0-9]*' )
	bf_files = [parameter_file]
	
	for folder in bf_folders:
		files = glob( folder )
		
		for file in files:
			
			bf_files.append(file)
	
	return bf_files

def get_avg_images(wildcards):
	
	import glob
	
	avg_images = glob.glob(OUTPUT_PATH + '/' + SAMPLE_NAME + '*_avg.hdf5')
	
	return avg_images

def centroids_3d_list(wildcards):
	
	import glob
	
	centroids_3d_file_list = glob.glob(OUTPUT_PATH + '/' + SAMPLE_NAME + '*' + 'centroids3d.txt')
	
	return centroids_3d_file_list
	
def get_bf_input( wildcards ):
	
	parameter_file = OUTPUT_PATH + '/' + SAMPLE_NAME + '_parameters.txt'
	bf_folders = glob( BRIGHTFIELD_FOLDER + '/Pos[0-9]*' )
	bf_files = [parameter_file]
	
	for folder in bf_folders:
		files = glob( folder )
		
		for file in files:
			
			bf_files.append(file)
	
	bf_files.append(METADATA_FILE)
	
	return bf_files

def onechannel_movie_input( wildcards ):
	
	if DRIFT:
		
		file = OUTPUT_PATH + '/' + SAMPLE_NAME +'_MaxZ_photobleachadj_driftadj.tif'
	else:
		file = OUTPUT_PATH + '/' + SAMPLE_NAME + '_MaxZ_photobleachadj.tif'
	
	return file

def overlay_movie_input( wildcards ):
	
	if DRIFT:
		
		file = OUTPUT_PATH + '/' + SAMPLE_NAME +'_MaxZ_photobleachadj_driftadj.tif'
	else:
		file = OUTPUT_PATH + '/' + SAMPLE_NAME + '_MaxZ_photobleachadj.tif'
	
	input = list([OUTPUT_PATH + '/' + SAMPLE_NAME + '_bf_stitched.tif', file])
	return input
	
	
	