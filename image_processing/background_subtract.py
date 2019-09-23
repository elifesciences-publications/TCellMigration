def background_subtract_image_segment( input_filename, drift_correction_trajectory_file, output_filename, output_nobgsub_filename, output_binary_filename, output_label_filename, output_parameters_filename, centroid_filename, lower_threshold=0, upper_threshold=100, binary_threshold=97):
	
	from skimage import io
	import numpy
	from skimage import filters, measure, morphology
	import scipy.stats
	
	print(lower_threshold, upper_threshold, binary_threshold)
	
	channel2_array = io.imread( input_filename )
	
	ntsteps, nx, ny = channel2_array.shape
	
	####Read in trajectory information
	
	trajectory_file = open(drift_correction_trajectory_file, 'r')
	trajectory = []
	for line in trajectory_file:
		line_list = line.strip().split('\t')[1].split(',')
		trajectory_coordinates = [float(s) for s in line_list]
		trajectory.append(trajectory_coordinates)
	trajectory_file.close()
	
	trajectory = numpy.array(trajectory)
	
	net_drift = numpy.cumsum(trajectory, axis = 0)
	

	lx_bdry = int(-1*numpy.min(net_drift[:,0]))
	ly_bdry = int(-1*numpy.min(net_drift[:,1]))
	
	rx_bdry = int(nx - numpy.max(net_drift[:,0]))
	ry_bdry = int(ny - numpy.max(net_drift[:,1]))
	
	####Exclude pixels that do not have data for part of the timeseries, due to drift
	
	
	nx_new = lx_bdry - rx_bdry
	
	ny_new = ly_bdry - ry_bdry
	
	file = open(output_parameters_filename, 'w')
	file.write("Drift mask bounderies\n")
	file.write("dimension 1," + str(lx_bdry) + "," + str(rx_bdry) + "\n")
	file.write("dimension 2," + str(ly_bdry) + "," + str(ry_bdry) + "\n")
	file.write("dimension 1 original boundaries," + "0," + str(nx) + "\n")
	file.write("dimension 2 original boundaries," + "0," + str(ny) + "\n")
	file.close()
	####
	
	channel2_array_smaller = channel2_array[:, lx_bdry:rx_bdry, ly_bdry:ry_bdry]
	
	io.imsave(output_nobgsub_filename, channel2_array_smaller)
	
	print(channel2_array_smaller.shape)
	
	####
	
	channel2_array_normed = numpy.zeros_like( channel2_array_smaller )
	
	binary_mask = numpy.zeros_like(channel2_array_smaller)
	object_array = numpy.zeros_like(channel2_array_smaller) ###Note that we have to create the object array at the start like this so that the object labels correspond to the labels for which we are recording centroid positions
	
	channel2_array_mean = numpy.mean( channel2_array_smaller, axis=0 )
	
	####
	centroid_file = open(centroid_filename, 'w')
	for t in range(ntsteps):
		if t%10 == 0:
			print('timestep',t)
			
		channel2_array_temp = channel2_array_smaller[t,:,:] - channel2_array_mean # + scipy.stats.mode(channel2_array_mean,axis=None)[0] ###Mean subtraction to get rid of constant nonspecific fluorescence and autofluorescence; add back the mode value which should be the background
		channel2_array_temp = numpy.maximum(channel2_array_temp, numpy.zeros_like(channel2_array_temp)) ###Clip at 0
		min_mat = numpy.percentile(channel2_array_temp, lower_threshold)*numpy.ones_like(channel2_array_smaller[t, :, :])
		
		max_mat = numpy.percentile(channel2_array_temp, upper_threshold)*numpy.ones_like(channel2_array_smaller[t, :, :])
		
		pix_range = max_mat - min_mat
		
		channel2_array_temp2 = numpy.minimum(channel2_array_temp, max_mat)
		channel2_array_normed[t,:,:] = numpy.maximum(channel2_array_temp2 - min_mat, numpy.zeros_like(channel2_array_temp2))/pix_range*4096.
		
		#####Segmentation via Sobel transform and watershed algorithm on normed, thresholded slices
		
		# edges = filters.sobel( channel2_array_normed[t,:,:] )
		# markers = numpy.zeros_like( channel2_array_normed[t,:,:] )
		# upper_marker_thresh = filters.threshold_otsu( channel2_array_normed[t,:,:] )
		
		# markers[ channel2_array_normed[t,:,:] < 30 ] = 1
		# markers[ channel2_array_normed[t,:,:] > upper_marker_thresh ] = 2
		
		# binary_slice = morphology.watershed( edges, markers ) - 1
		
		#Remove very small objects (usually pixel noise)
		
		# objects_all = scipy.ndimage.measurements.label(binary_slice)[0]
		
		# objects_largish = morphology.remove_small_objects( objects_all, 8 )
		
		#Remove very large objects
		
		# objects_filtered = numpy.zeros_like(objects_largish)
		
		# centroid_strs = []
		
		#####
		
		# binary_slice_filtered = numpy.zeros_like(binary_slice)
		
		# objects_filtered = numpy.zeros_like(objects_largish)
		
		# region_props = measure.regionprops(objects_largish, channel2_array_normed[t,:,:])
		# centroid_strs = []
		
		# for reg in region_props:
			# mass = reg.moments[0,0]*reg.area
			# bbox_coords = reg.bbox
			# ax1_len = bbox_coords[2] - bbox_coords[0]
			# ax2_len = bbox_coords[3] - bbox_coords[1]
			# aspect_ratio = max(ax1_len,ax2_len)/min(ax1_len,ax2_len)
			
			# if ((reg.area < 2000 and mass > 4000) and (aspect_ratio < 10)):
				
				
				# centroid_str = str(reg.label) + ',' + str(reg.weighted_centroid) + ',' + str(mass)
				# centroid_strs.append(centroid_str)
				# objects_filtered[numpy.where(objects_largish==reg.label)] = reg.label
				# binary_slice_filtered[numpy.where(objects_largish==reg.label)] = 1
			
			
		#centroid_line = ('\t').join( centroid_strs ) + '\n'
		
		#centroid_file.write(centroid_line)
		
		#binary_mask[t,:,:] = binary_slice_filtered
		#object_array[t,:,:] = objects_filtered
		
	#print('finished the loop')
	centroid_file.close()
	
	data_array = numpy.array(channel2_array_normed, dtype='uint16')
	io.imsave(output_filename, data_array)
	
	####Save binary masks for each timestep, for objects that passed filter; also save the object labels
	
	io.imsave(output_binary_filename, binary_mask)
	io.imsave(output_label_filename, object_array)
	
	
if __name__=="__main__":
	
	import sys
	input_filename = sys.argv[1]
	drift_correction_trajectory_file = sys.argv[2]
	output_filename = sys.argv[3]
	output_nobgsub_filename = sys.argv[4]
	output_binary_filename = sys.argv[5]
	output_label_filename = sys.argv[6]
	output_parameter_filename = sys.argv[7]
	centroid_filename = sys.argv[8]
	lower_threshold = 0
	upper_threshold = 100
	binary_threshold = 97
	
	if len(sys.argv) > 9:
		lower_threshold = float(sys.argv[9])
	if len(sys.argv) > 10:
		upper_threshold = float(sys.argv[10])
	if len(sys.argv) > 11:
		binary_threshold = float(sys.argv[11])
	
	background_subtract_image_segment( input_filename, drift_correction_trajectory_file, output_filename, output_nobgsub_filename, output_binary_filename, output_label_filename, output_parameter_filename, centroid_filename, lower_threshold, upper_threshold, binary_threshold)