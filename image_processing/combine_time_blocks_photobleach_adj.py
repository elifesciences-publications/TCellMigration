def combine_time_blocks( file_list, percentile = 99.8, max_ntsteps = 1000 ):
	
	###This is a utility file to concatenate multiple .hdf5 files into a .tif stack. Concatenation occurs along axis 0 (usually time)
	
	import numpy
	import h5py
	from skimage import io
	
	###
	h5file = h5py.File( file_list[0], 'r')
	
	dset = h5file["xytstack"] ###TO DO generalize to other dset names
	
	darray = numpy.array(dset)
	
	h5file.close()
	###
	
	for file in file_list[1:]:
		
		h5file = h5py.File( file, 'r')
		
		dset = h5file["xytstack"] ###TO DO generalize to other dset namesj
		
		darray = numpy.concatenate((darray, dset), axis = 0)
		
		h5file.close()
	
	###Photobleaching adjustment
	
	ntsteps_tot, nx, ny = darray.shape
	print(max_ntsteps)
	ntsteps = min(ntsteps_tot, max_ntsteps)
	
	darray_normed = numpy.zeros_like(darray[0:ntsteps,:,:])
	
	for t in range(ntsteps):
		drift_mask = numpy.nonzero( darray[t, :, :] )
		norm_mat = numpy.percentile(darray[t, :, :][drift_mask], percentile )*numpy.ones_like(darray[t, :, :])
		darray_normed[t,:,:] = numpy.minimum(darray[t, :, :], norm_mat)/numpy.percentile( darray[t, :, :], percentile )*4096.
		
	###Cast to uint16
	
	darray_normed = numpy.array(darray_normed, dtype='uint16')
	
	###Save files as .tif
	
	output_filename1 = file_list[0].strip('_block0_MaxZ_driftcorr.hdf5') + ('_MaxZ.tif')
	
	io.imsave(output_filename1, darray[0:ntsteps,:,:])
	
	output_filename2 = file_list[0].strip('_block0_MaxZ_driftcorr.hdf5') + ('_MaxZ_photobleachadj.tif')
	
	io.imsave(output_filename2, darray_normed)
	
if __name__=="__main__":
	
	import sys
	
	file_list = sys.argv[1:-2]
	
	percentile = float(sys.argv[-2])
	
	max_ntsteps = int(sys.argv[-1])
	
	combine_time_blocks( file_list, percentile, max_ntsteps )