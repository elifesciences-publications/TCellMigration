def stitch_bf( image_parent_directory, metadata_file, fluor_metadata_file, output_name, parameter_file, overlap_averaging=True ):
	
	from skimage import io
	import numpy
	import h5py
	import utilities2
	import glob
	
	###This function constructs and saves a 4D stack from individual .tif images, as an .hdf5 file. Note that tiles are stitched together according to the locations listed in position_list_file.
	###Note also that the final stack will be the smallest bounding rectangular prism for the acquisition volume, with zeros filled in where there is no data. (For future consideration: maybe a placeholder inaccessible value is better so that it's easy to clearly plot?)
	
	###Inputs:
	###Image_parent_directory: Directory containing the images. Expected to contain a subfolder for each position in position_list_file, which contains images that form a timeseries.
	###Position_list_file: A file with the stage coordinates for each recorded position. These files are in the format saved by micromanager, and are text files with a .pos extension.
	###output_name: file name for .tif output file.
	###fluor_tif_file: the file that will be overlaid on this brightfield image in the next step; used to get the pixel dimensions
	
	x_fov = 571.28 ###Dimensions of each field of view in microns
	y_fov = 763.7
	
	###Import pixel dimensions from parameter file
	
	file = open(parameter_file,'r')
	lines = file.readlines()
	file.close()
	linelist = lines[1].strip().split(',')
	xbounds = [int(l) for l in linelist[1:]]
	linelist = lines[2].strip().split(',')
	ybounds = [int(l) for l in linelist[1:]]
	linelist = lines[3].strip().split(',')
	nx_orig_fluor = int(linelist[2])
	linelist = lines[4].strip().split(',')
	ny_orig_fluor = int(linelist[2])
	
	###Get a dict of the stage positions associated with each tile, organized by slice
	
	slice_dict, reverse_loc_dict = utilities2.get_positions4( metadata_file )
	
	slice_dict_fluor, reverse_loc_dict_fluor = utilities2.get_positions4( fluor_metadata_file )
	
	loc_dict = {}
	
	pos_set = set()
	
	for key,val in reverse_loc_dict.items():
		loc_dict[val] = key
		pos,slice = val
		pos_set.add(pos)
	
	loc_dict_fluor = {}
	
	for key,val in reverse_loc_dict_fluor.items():
		loc_dict_fluor[val] = key

	
	###Get a list of folders (positions) in the parent dir
	
	folders = glob.glob( image_parent_directory + '/Pos[0-9]*' )
	
	metadata_dict = utilities2.get_stack_metadata( metadata_file )
	
	###Extract pixel dimensions from metadata
	
	pixels = int( metadata_dict[ "Width" ] )
	#print(pixels)
	pixels_h = int( metadata_dict[ "Height" ] )
	#print(pixels_h)
	pixel_width = y_fov/float(pixels)
	
	print(pixel_width)
	
	###Get bounding box (dimensions in microns, as determined by pos_list_file list of stage positions)
	
	[(xmin,xmax),(ymin,ymax),(zmin,zmax)] = utilities2.get_bounding_box( reverse_loc_dict )
	
	#print("Dimensions", (xmin,xmax),(ymin,ymax),(zmin,zmax) )
	
	pixel_dim1 = int(numpy.ceil((ymax - ymin)/pixel_width)) + pixels
	
	pixel_dim2 = int(numpy.ceil((zmax - zmin)/pixel_width)) + pixels_h
	
	print(pixel_dim1, pixel_dim2)
	
	pixel_dim2_fluor = xbounds[1]-xbounds[0]
	pixel_dim1_fluor = ybounds[1]-ybounds[0]
	###
	
	pixel_ratio1 = pixel_dim1/pixel_dim1_fluor
	pixel_ratio2 = pixel_dim2/pixel_dim2_fluor
	
	###
	
	bin = False
	
	if ((pixel_ratio1 > .9 and pixel_ratio1 < 1.2) and (pixel_ratio2 > .9 and pixel_ratio2 < 1.2)):
		####Binning was the same in fluor and bf images; make the array the same size for both
		
		stitched_bf_array = numpy.zeros((nx_orig_fluor, ny_orig_fluor), dtype='uint16')
	
	elif ((pixel_ratio1 > 1.6) and (pixel_ratio2 > 1.6)):
		####Binning was 2x2 in fluor and 1x1 in bf images
		stitched_bf_array = numpy.zeros((2*nx_orig_fluor, 2*ny_orig_fluor), dtype='uint16')
		stitched_bf_array_binned = numpy.zeros((nx_orig_fluor, ny_orig_fluor), dtype='uint16')
		bin = True
	
	else:
		print(pixel_ratio1, pixel_ratio2)
		raise RuntimeError("You are specifying the dimensions wrong!")
	
	print(stitched_bf_array.shape)
	for folder in folders:
		
		pos_name = folder.split('/')[-1]
		if pos_name in pos_set:
			slice_file = glob.glob( image_parent_directory + '/' + pos_name + '/img_000000000*.tif' )[0]
			
			if slice_file.split('_')[-1] == '000.tif':
				slice = 0
			else:
				slice = int(slice_file.split('_')[-1].strip('0.tif'))
			
			loc = loc_dict[(pos_name,slice)]
			
			ystart_pix = int(numpy.ceil((loc[1] - ymin)/pixel_width))
			
			fluorlocx, fluorlocy, fluorlocz = loc_dict_fluor[(pos_name,0)]
			
			fluory_start_pix = int(numpy.ceil((fluorlocy - ymin)/pixel_width))
			print(fluory_start_pix,ystart_pix)
			
			#print(ystart_pix, ystart_pix + pixels)
			zstart_pix = int(numpy.ceil((loc[2] - zmin)/pixel_width))
			#print(zstart_pix, zstart_pix + pixels_h)
			
			if overlap_averaging:
				new_data = numpy.zeros_like(stitched_bf_array)
				nz,ny = new_data.shape
				if fluory_start_pix == ystart_pix:
					new_data[zstart_pix:zstart_pix + pixels_h, ystart_pix:ystart_pix + pixels] = numpy.flipud(io.imread( slice_file ))
				elif fluory_start_pix > ystart_pix:
				
					new_data[zstart_pix:zstart_pix + pixels_h, fluory_start_pix:] = numpy.flipud(io.imread( slice_file )[:, 0:ny - fluory_start_pix])
				else:
					new_data[zstart_pix:zstart_pix + pixels_h, ystart_pix:fluory_start_pix + pixels] = numpy.flipud(io.imread( slice_file )[:, 0:fluory_start_pix + pixels - ystart_pix])
					
				overlap_region = numpy.array((stitched_bf_array>0)*(new_data>0), dtype='bool')
				
				stitched_bf_array += new_data
				stitched_bf_array[overlap_region] = stitched_bf_array[overlap_region]/2
				
				
			else:
				stitched_bf_array[zstart_pix:zstart_pix + pixels_h, ystart_pix:ystart_pix + pixels] = numpy.flipud(io.imread( slice_file ) )
		
		if bin:
			for i in range(pixel_dim2_fluor):
				for j in range(pixel_dim1_fluor):
					stitched_bf_array_binned[i,j] = numpy.mean( [stitched_bf_array[2*i,2*j], stitched_bf_array[2*i+1,2*j], stitched_bf_array[2*i,2*j+1], stitched_bf_array[2*i+1,2*j+1]] )
			
			stitched_bf_array_binned_frame = stitched_bf_array_binned[xbounds[0]:xbounds[1], ybounds[0]:ybounds[1]]
			io.imsave(output_filename, stitched_bf_array_binned_frame)
		else:
			stitched_bf_array_frame = stitched_bf_array[xbounds[0]:xbounds[1], ybounds[0]:ybounds[1]]
			io.imsave(output_filename, stitched_bf_array_frame)

if __name__=="__main__":
	
	import sys
	
	input_folder = sys.argv[1]
	metadata_file = sys.argv[2]
	output_filename = sys.argv[3]
	parameter_file = sys.argv[4]
	fluor_metadata_file = sys.argv[5]
	
	stitch_bf(input_folder, metadata_file, fluor_metadata_file, output_filename, parameter_file)
	
	