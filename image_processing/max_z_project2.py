def max_z_project( input_hdf5_file ):
	
	###This function takes a 4D dataset (txyz) and takes a maximum z projection at each timestep.
	
	###Input: .hdf5 file; the only dataset should be a 4D image stack, with dimensions ordered txyz, where t is time and z is the dimension we are collapsing
	###Output: .hdf5 file; will have the same name as the input, with _MaxZ.hdf5 appended.
	
	import numpy
	import h5py
	
	h5file = h5py.File( input_hdf5_file, 'r' )
	
	dset = h5file["xyztstack"]
	
	ntsteps, pixel_dim2_adj, pixel_dim1, z = dset.shape
	
	####Initialize the new hdf5 for writing
	
	if 'block5' not in input_hdf5_file:
		output_filename = input_hdf5_file.strip('_raw.hdf5') + '_MaxZ.hdf5'
	else:
		output_filename = input_hdf5_file.strip('_raw.hdf5') + '5_MaxZ.hdf5'
		
	h5file_output = h5py.File(output_filename, 'w')
	
	MB_per_slice = 16.*pixel_dim1*pixel_dim2_adj/8./10**6
	
	chunks_per_slice = int( numpy.ceil( MB_per_slice ) )
	
	dset_for_writing = h5file_output.create_dataset("xytstack", (ntsteps, pixel_dim2_adj, pixel_dim1), dtype='uint16', chunks=(1,pixel_dim2_adj/chunks_per_slice,pixel_dim1))
	
	#####
	
	for t in range(ntsteps):
		
		xyzvol = dset[t, :, :, :]
		
		dset_for_writing[t, :, :] = numpy.max( xyzvol, axis = 2 )
		
	
	h5file_output.close()
	
	if 'block5' not in input_hdf5_file:
		output_filename_slice0 = input_hdf5_file.strip('_raw.hdf5') + '_MaxZ_slice0.hdf5'
	else:
		output_filename_slice0 = input_hdf5_file.strip('_raw.hdf5') + '5_MaxZ_slice0.hdf5'
		
	h5file_slice0 = h5py.File(output_filename_slice0, 'w')
	
	slice0_data = numpy.max( dset[0, :, :, :], axis = 2 )
	
	dset_slice0 = h5file_slice0.create_dataset("slice0", data=slice0_data)
	
	h5file_slice0.close()
	h5file.close()
if __name__=="__main__":
	
	import sys
	
	hdf5_input = sys.argv[1]
	max_z_project(hdf5_input)