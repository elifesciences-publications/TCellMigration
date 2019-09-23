def stitch_tiles( image_parent_directory, metadata_file, file_saving_dir, tblock_name, max_tsteps ):
	
	from skimage import io
	import numpy
	import h5py
	import utilities2
	
	###This version of stitch_tiles is for movies saved with explicit z-slices; the zt stack at a particular frame is in one folder.
	###This function constructs and saves a 4D stack from individual .tif images, as an .hdf5 file. Note that tiles are stitched together according to the locations listed in position_list_file.
	###Note also that the final stack will be the smallest bounding rectangular prism for the acquisition volume, with zeros filled in where there is no data. (For future consideration: maybe a placeholder inaccessible value is better so that it's easy to clearly plot?)
	
	###Inputs:
	###Image_parent_directory: Directory containing the images. Expected to contain a subfolder for each position in position_list_file, which contains images that form a timeseries.
	###Position_list_file: A file with the stage coordinates for each recorded position. These files are in the format saved by micromanager, and are text files with a .pos extension.
	###file_saving_dir: directory where the final .hdf5 will be saved.
	###tblock: a list of ints specifying which timesteps should be processed. (Intended for parallelization)
	
	x_fov = 571.28 ###Dimensions of each field of view in microns
	y_fov = 763.7
	
	###Get a dict of the stage positions associated with each tile, organized by slice
	
	slice_dict, reverse_loc_dict = utilities2.get_positions4( metadata_file )
	
	#print(slice_dict)
	
	###Go to Metadata file in one of the image file folders and grab acquisition properties
	
	metadata_dict = utilities2.get_stack_metadata( metadata_file )
	
	trange_dict = utilities2.get_tranges(metadata_dict, max_tsteps)
	
	trange_strs = trange_dict[tblock_name]
	
	
	trange = [int(t) for t in trange_strs.strip(',').split(',')]
	
	print(tblock_name, trange)
	#print(metadata_dict)
	
	###Extract number of timesteps and pixel dimensions from metadata
	
	ntsteps = int( metadata_dict[ "Frames" ] )
	
	pixels = int( metadata_dict[ "Width" ] )
	#print(pixels)
	pixels_h = int( metadata_dict[ "Height" ] )
	#print(pixels_h)
	pixel_width = y_fov/float(pixels)
	
	#print(pixel_width)
	
	###Get bounding box (dimensions in microns, as determined by pos_list_file list of stage positions)
	
	[(xmin,xmax),(ymin,ymax),(zmin,zmax)] = utilities2.get_bounding_box( reverse_loc_dict )
	
	#print("Dimensions", (xmin,xmax),(ymin,ymax),(zmin,zmax) )
	
	pixel_dim1 = int(numpy.ceil((ymax - ymin)/pixel_width)) + pixels
	
	pixel_dim2 = int(numpy.ceil((zmax - zmin)/pixel_width)) + pixels_h
	
	nslices = len(slice_dict.keys())
	
	#print("The width and height in pixels", pixel_dim1, pixel_dim2)
	###Initialize the hdf5 file that will be used to store the final stack
	
	fish_name = image_parent_directory.split('/')[-1]
	
	output_filename = file_saving_dir + '/' + fish_name + '_' + tblock_name + '_raw.hdf5'
	
	f = h5py.File(output_filename, 'w')
	
	###Determine the chunk structure based on the guideline that each chunk should be slightly less than 1 MiB (parameter for storage and io of hdf5 file)
	
	MB_per_slice = 16.*pixel_dim1*pixel_dim2/8./10**6
	
	chunks_per_slice = int( numpy.ceil( MB_per_slice ) )
	
	pixel_dim2_adj = pixel_dim2 + pixel_dim2 % chunks_per_slice ###Add pixels if necessary so that chunks per slice divides evenly into the long axis (this should only be a few rows of pixels, since chunks_per_slice should be ~4-10, or approximately the number of tiles per slice)
	
	dset = f.create_dataset("xyztstack", (1, pixel_dim2_adj, pixel_dim1, nslices), dtype='uint16', maxshape = (ntsteps, pixel_dim2_adj, pixel_dim1, nslices), chunks=(1,pixel_dim2_adj/chunks_per_slice,pixel_dim1,1))
	
	###Put the metadata into the data set
	
	for property in metadata_dict:
		
		dset.attrs[property] = metadata_dict[property]
		
	###Construct the stack, timestep by timestep
	
	###Get the timesteps we're using this round
	
	# trange = []
	# for name in files_in_trange:
		# t = files_in_trange.split('/')[-1].split('_')[1]
		# trange.append( t )
	
	t_ind = 0
	for t in trange:
		
		slice_ind = 0
		
		tstr = str(t)
		
		dset.resize(t_ind+1, axis = 0)
		
		while len(tstr) < 8.5:
		
			tstr = '0' + tstr
							
		for slice in slice_dict:
			
			data = numpy.zeros((pixel_dim2_adj, pixel_dim1), dtype = 'uint16')
			
			for loc in slice_dict[slice]:
				
				pos_name, slice_index = reverse_loc_dict[loc]
				
				slice_str = str(slice_index)
				
				while len(slice_str) < 2.5:
					slice_str = '0' + slice_str
					
				slice_file = image_parent_directory + '/' + pos_name + '/img_' + tstr + '_Default_' + slice_str + '.tif'
				
				ystart_pix = int(numpy.ceil((loc[1] - ymin)/pixel_width))
				#print(ystart_pix, ystart_pix + pixels)
				zstart_pix = int(numpy.ceil((loc[2] - zmin)/pixel_width))
				#print(zstart_pix, zstart_pix + pixels_h)
				
				data[zstart_pix:zstart_pix + pixels_h, ystart_pix:ystart_pix + pixels] = numpy.flipud(io.imread( slice_file )) ###Note that we are, for the moment, including only one representation of any overlap regions
			
			
			dset[t_ind,:,:,slice_ind] = data
			
			slice_ind += 1
		t_ind += 1
		#io.imsave( output_tifname + '_' + str(t) +'.tif', dset[t,:,:,:].T )
		
		if t % 10 == 0:
			print("finished with timestep ", t)
	f.close()
	
	return 'Done'

if __name__=="__main__":
	
	import sys
	
	input_folder = sys.argv[1]
	pos_list = sys.argv[2]
	output_folder = sys.argv[3]
	tblock_name = sys.argv[4]
	max_tsteps = int(sys.argv[5])
	
	done = stitch_tiles(input_folder, pos_list, output_folder, tblock_name, max_tsteps)
	
	