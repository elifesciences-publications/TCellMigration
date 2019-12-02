def max_z_project( input_hdf5_file, output_tif_filename ):
	
	###This function takes a 4D dataset (txyz) and takes a maximum z projection at each timestep.
	
	###Input: .hdf5 file; the only dataset should be a 4D image stack, with dimensions ordered txyz, where t is time and z is the dimension we are collapsing
	###Output: .hdf5 file; will have the same name as the input, with _MaxZ.hdf5 appended.
	
	import numpy
	import h5py
	from skimage import io
	
	h5file = h5py.File( input_hdf5_file, 'r' )
	
	dset = h5file["xyztstack"]
	
	ntsteps, pixel_dim2_adj, pixel_dim1, z = dset.shape
	
	io.imsave(output_tif_filename,dset[0,:,:,:].swapaxes(0,2))
	
	h5file.close()
if __name__=="__main__":
	
	import sys
	
	hdf5_input = sys.argv[1]
	output_tif = sys.argv[2]
	max_z_project(hdf5_input,output_tif)