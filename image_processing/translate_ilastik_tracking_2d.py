def translate_ilastik_tracking( maxz_filename, mask_filename, merger_filename, output_filename, tstart, tstop):
	import numpy
	from glob import glob
	from skimage import io, measure
	from scipy.ndimage import measurements
	from scipy.stats import moment
	import h5py
	
	mask_file = h5py.File( mask_filename, 'r')
	print("Read in mask file")
	tabs0 = tstart
	tabsend = tstop
	
	mask_array = numpy.array( mask_file['exported_data'][tabs0:tabsend,:,:,0,0] )
	
	merger_file = open(merger_filename,'r')
	merger_dict = {}
	firstline = True
	for line in merger_file:
		if firstline:
			firstline=False
		else:
			linelist = line.strip().split(',')
			if len(linelist) > 1:
				timestep = int(linelist[0])
				ids = [int(s) for s in linelist[2].split(';')]
				
				if timestep not in merger_dict:
					merger_dict[timestep] = []
				
				merger_dict[timestep].append(ids)
	
	merger_file.close()
		
	#mask_array = io.imread( mask_filename )
	
	ntsteps_mask, nx_mask, ny_mask = mask_array.shape
	
	hdf5_file_counter = 0
	
	moment2zs = []
	
	dset = io.imread( maxz_filename )
			
	ntstepsh5, nxh5, nyh5 = dset.shape
	print(dset.shape)
	print(mask_array.shape)
	#####
	
	#extended_mask = numpy.zeros_like( dset, dtype='uint8' )
	
	

	track_dict = {}
	tracks_file = open(output_filename, 'w')
	
	for t in range(tstop-tstart):
		
		#print(max(mask_array[tabs0 + t, :, :].flatten()))
		cell_index_set = set( mask_array[t, :, :].flatten())
		cell_index_set.remove(0) ###This is the background, not a cell
		cell_indexes = list( cell_index_set )
		
		
		coms = measurements.center_of_mass( dset[tabs0 + t,:,:], mask_array[t, :, :].swapaxes(0,1), cell_indexes  )
		
		for i in range(len(coms)):
			com = coms[i]
			
			txyz = (tabs0 + t,com[0],com[1],0)
			track_ind = cell_indexes[i]
			if track_ind not in track_dict:
				track_dict[track_ind] = []
				
			track_dict[track_ind].append( txyz )
		
		###If there are any mergers on this timestep, only one of the tracks will have been assigned the centroid. Go through and copy over this centroid position to the merged objects.
		
		if tabs0 + t in merger_dict:
		
			print("handling mergers")
			mergers = merger_dict[tabs0 + t]
			
			for merger in mergers:
				
				found_txyz = False
				
				for track_ind in merger:
					###Figure out which track was associated with this object at this timestep in the segmented images, and get the centroid info from that track
					if track_ind in track_dict:
						most_recent_txyz = track_dict[track_ind][-1]
						if most_recent_txyz[0] == tabs0 + t:
							
							if found_txyz:
								print("something is wrong; more than one of these tracks was assigned!")
								raise ValueError()
								
							txyz_to_copy = most_recent_txyz
							already_assigned = track_ind
							
							found_txyz = True
				
				if not found_txyz:
					print("Something is wrong; none of these tracks were assigned!")
					raise ValueError()
					
				for track_ind in merger:
					
					if track_ind != already_assigned:
						
						if track_ind not in track_dict:
							track_dict[track_ind] = []
						
						track_dict[track_ind].append( txyz_to_copy )
		
		
	for ind in track_dict:
		track = track_dict[ind]
		output_strs = []
		for entry in track:
			output_list = [str(s) for s in entry]
			output_strs.append( (',').join(output_list) )
		tracks_file.write(str(ind) + '\t' + ('\t').join( output_strs ) + '\n')
	
	tracks_file.close()
	
if __name__=='__main__':
	
	import sys
	
	maxz_filename = sys.argv[1]
	mask_filename = sys.argv[2]
	merger_filename = sys.argv[3]
	output_filename = sys.argv[4]
	tstart = int(sys.argv[5])
	tstop = int(sys.argv[6])
	
	translate_ilastik_tracking( maxz_filename, mask_filename, merger_filename, output_filename, tstart, tstop)