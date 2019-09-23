def drift_adjustment( image_file, anchor_centers, output_filename, drift_trajectory_file, tracking = True, region_radius=50, ref_tstep=0 ):
	
	###This function tracks an *actually stationary* object to account for drift in the x-y plane. This is only designed to work in the simplest case, where there is an isolated stationary object to use as an anchor (for example a pigment spot)

	###Parameters:
	###input_array: a numpy array containing input images, assumed to be (Txmxn)
	###anchor_center: the center of the stationary object in the reference timestep, which is either the first or last frame. Can be approximate; the function will recalculate the centroid.
	###region_radius: the radius of the region over which to calculate the centroid in the next frame.
	###ref_tstep: drift displacement vectors will be calculated relative to the location of the object at this timestep. Default is 0.
	
	###Calculate trajectory of stationary object by finding the brightness centroid within the circle with r=region_radius, around the center from the previous frame.
	
	import numpy
	from scipy.ndimage.measurements import center_of_mass, label
	import matplotlib.pylab as pt
	from skimage import io, filters, morphology
	
	image_data = io.imread( image_file )
	

	ntsteps = image_data.shape[0]
	
	if not tracking_flag:
		
		output_filename = io.imsave( output_filename, image_data )
		
		file = open(drift_trajectory_file, 'w')
	
		for t in range(ntsteps):
			file.write(str(t) + '\t' + '0,0' + '\n')
		file.close()
	
	else:
		coms = anchor_centers
		
		circle_mask = numpy.zeros((2*region_radius+1, 2*region_radius+1), dtype='int')
		
		for j in range(region_radius):
			
			x = numpy.sqrt( (region_radius)**2 - j**2 )
			circle_start = region_radius - int(round(x))
			circle_end = region_radius + int(round(x))
			
			for i in range(circle_start,circle_end + 1):
				
				circle_mask[i,region_radius - j] = 1
				circle_mask[i, region_radius + j] = 1
				
		com_trajectories = {}
		for com in coms:
			com_trajectories[tuple(com)] = []
		com_trajectories['avg_disp'] = []
		#print(coms)
		#ntsteps = 10
		for i in range(ntsteps):
			
			if ref_tstep == -1: ###Trajectory will be calculated from the final tstep backwards
				
				t = ntsteps - i - 1
			else:
				t = i
			
			disps = numpy.array([0.,0.])
			
			for com in coms:
				
				com_list = com_trajectories[tuple(com)]
				if len(com_list) < 1:
					prev_com = com
				else:
					prev_com = com_list[i-1]
				
				region = image_data[t,int(round(prev_com[0]))-region_radius:int(round(prev_com[0]))+region_radius+1,int(round(prev_com[1]))-region_radius:int(round(prev_com[1]))+region_radius+1]#*circle_mask
				
				###Find the spot in the region
				
				otsu_thresh = filters.threshold_otsu( region )
				mask = numpy.zeros_like(region)
				mask[region>otsu_thresh] = 1
				labels, num_features = label(mask)
				
				large_label = morphology.remove_small_objects(labels,min_size=4)
				
				###Find the com of the labeled region
				
				
				local_com = center_of_mass( region, labels=large_label )
				
				#pt.pcolor(region)
				#pt.axis('equal')
				#pt.plot([local_com[1]],[local_com[0]], marker='o',color='k')
				
				
				#print(local_com)
				new_com = [int(round(prev_com[0])) - region_radius + local_com[0], int(round(prev_com[1])) - region_radius + local_com[1]]
				
				#pt.figure(figsize=(8,20))
				#pt.pcolor(image_data[t,:,:])
				#pt.plot([new_com[1]], [new_com[0]], marker='o',color='k')
				#pt.show()
				
				if t == 0:
					disp = numpy.array([0.,0.])
				else:
					disp = numpy.array(new_com) - numpy.array(prev_com)
				
				com_trajectories[tuple(com)].append(new_com)
				
				disps += disp
				
				#print(local_com[0])
				#print(prev_com[0])
				#print(int(round(prev_com[0])) - region_radius)
				#print(new_com[0])
				#print(disps)
		
			com_trajectories['avg_disp'].append( disps/2. )
			#print(disp)
		
		drift_corrected_array = numpy.zeros_like( image_data )
		
		trajectories = numpy.array(com_trajectories['avg_disp'])
		for i in range(ntsteps):
			
			driftx = numpy.int(numpy.round(numpy.sum(trajectories[0:i+1,:],axis=0)[0]))
			drifty = numpy.int(numpy.round(numpy.sum(trajectories[0:i+1,:],axis=0)[1]))
			
			print(driftx, drifty)
			if ref_tstep == -1: ###Trajectory will be calculated from the final tstep backwards
				
				t = ntsteps - i - 1
			else:
				t = i
				
			if (driftx >= 0 and drifty >= 0):
				if (driftx == 0 and drifty == 0):
					drift_corrected_array[t, :, :] = image_data[t, :, :]
				elif driftx == 0:
					drift_corrected_array[t, :, :-drifty] = image_data[t, :, drifty:]
				elif drifty == 0:
					drift_corrected_array[t, :-driftx, :] = image_data[t, driftx:, drifty:]
				else:
					drift_corrected_array[t, :-driftx, :-drifty] = image_data[t, driftx:, drifty:]
			
			elif (driftx >= 0 and drifty < 0):
				if driftx == 0:
					drift_corrected_array[t, :, -drifty:] = image_data[t, driftx:, :drifty]
				else:
					drift_corrected_array[t, :-driftx, -drifty:] = image_data[t, driftx:, :drifty]
			
			elif (driftx < 0 and drifty >= 0):	
				if drifty == 0:
					drift_corrected_array[t, -driftx:, :] = image_data[t, :driftx, drifty:]
				else:
					drift_corrected_array[t, -driftx:, :-drifty] = image_data[t, :driftx, drifty:]
			
			elif (driftx < 0 and drifty < 0):
				
				drift_corrected_array[t, -driftx:, -drifty:] = image_data[t, :driftx, :drifty]
			
			
				
		io.imsave( output_filename, drift_corrected_array )
		
		file = open(drift_trajectory_file, 'w')
		
		for t in range(ntsteps):
			file.write(str(t) + '\t' + (',').join([str(d) for d in com_trajectories['avg_disp'][t]]) + '\n')
		file.close()
	

if __name__=='__main__':
	
	import sys
	
	image_file = sys.argv[1]
	output_filename = sys.argv[2]
	drift_trajectory_file = sys.argv[3]
	radius = int(sys.argv[4])
	
	if len(sys.argv) > 7:
		tracking_x1 = int(sys.argv[5])
		tracking_y1 = int(sys.argv[6])
		tracking_x2 = int(sys.argv[7])
		tracking_y2 = int(sys.argv[8])
		tracking_flag = True
		drift_adjustment( image_file, [[tracking_y1,tracking_x1],[tracking_y2,tracking_x2]], output_filename, drift_trajectory_file, region_radius=radius, tracking = tracking_flag )
	elif len(sys.argv) > 6:
		tracking_x1 = int(sys.argv[5])
		tracking_y1 = int(sys.argv[6])
		tracking_flag = True
		drift_adjustment( image_file, [[tracking_y1,tracking_x1]], output_filename, drift_trajectory_file, region_radius=radius, tracking = tracking_flag )
	else:
		tracking_x1 = 0
		tracking_y1 = 0
		tracking_x2 = 0
		tracking_y2 = 0
		tracking_flag = False
		
		drift_adjustment( image_file, [[tracking_y1,tracking_x1],[tracking_y2,tracking_x2]], output_filename, drift_trajectory_file, region_radius=radius, tracking = tracking_flag )
	###(180,1047)