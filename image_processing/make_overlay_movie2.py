def make_overlay_movie( channel1_file, channel2_file, outfile_name, mask='None', bg_thresh=.4, color1=(1.,1.,1.), color2=(0.,1.,0.), mixing_ratio=.5, lower_threshold=0, upper_threshold=100., maxtsteps=1000 ):
	
	###This function overlays image(s) from channel1 and channel2, and saves the resulting movie.
	
	###Parameters:
	###channel1_file: .tif file containing image(s) from channel1 (by default this is the grayscale/bf channel). Note that channel1 can be a still (single frame); if this is the case then it will be used at each slice.
	###channel2_file: .tif file containing image(s) from channel2 (default color green).
	###outfile_name: output file name. Should be .mp4.
	###color1: (R,G,B) color for channel1. Default (1,1,1) (grayscale)
	###color2: (R,G,B) color for channel2. Default (0,1,0) (green)
	###mask: either 'None' or a map with predictions of whether a pixel in each channel2 image is foreground or background. Given that channel2_array is (TxMxN), mask is expected to be (TxMxNx2). If a mask is included, pixels from channel2 with foreground probability less than bg_thresh will not be overlaid.
	###bg_thresh: only used with 'mask'. The probability threshold above which a pixel will be classified as foreground.
	###lower_threshold: pixel values will be thresholded frame-by-frame at the percentile chosen as 'lower_threshold'. Default is 0.
	
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pylab as pt
	import matplotlib.animation as manimation
	import numpy
	from skimage import io
	
	channel1_array = io.imread(channel1_file)
	channel2_array = io.imread(channel2_file)
	
	ntsteps, nx, ny = channel2_array.shape

	output_matrix = numpy.zeros((ntsteps, nx, ny, 3), dtype = "float")
	
	if mask != 'None':
		foreground_prob = mask[:,:,:,0]
		#foreground_prob = numpy.swapaxes( mask[:,:,:,0], 1, 2)
		
	else:
	
		foreground_prob = numpy.ones_like( channel2_array ) ###If mask == 'None', all pixels will be included
		
	if channel2_array.shape == channel1_array.shape:
		
		all_T_slices = True
	
	else:
		
		all_T_slices = False ###channel1 is a still
	
	channel2_array_normed = numpy.zeros_like( channel2_array )
	
	for t in range(ntsteps):
		
		min_mat = numpy.percentile(channel2_array[t, :, :], lower_threshold)*numpy.ones_like(channel2_array[t, :, :])
		
		max_mat = numpy.percentile(channel2_array[t, :, :], upper_threshold)*numpy.ones_like(channel2_array[t, :, :])
		
		pix_range = max_mat - min_mat
		
		channel2_array_temp = numpy.minimum(channel2_array[t, :, :], max_mat)
		
		channel2_array_normed[t,:,:] = numpy.maximum(channel2_array_temp - min_mat, numpy.zeros_like(channel2_array_temp))/pix_range*4096
		
		if not all_T_slices:
		
			output_matrix[t, :, :, 0] += channel1_array*mixing_ratio*color1[0]
			output_matrix[t, :, :, 1] += channel1_array*mixing_ratio*color1[1]
			output_matrix[t, :, :, 2] += channel1_array*mixing_ratio*color1[2]
		
		else:
			
			output_matrix[t, :, :, 0] += channel1_array[t, :, :]*mixing_ratio*color1[0]
			output_matrix[t, :, :, 1] += channel1_array[t, :, :]*mixing_ratio*color1[1]
			output_matrix[t, :, :, 2] += channel1_array[t, :, :]*mixing_ratio*color1[2]
		
		utility_zeros = numpy.zeros_like(channel2_array_normed[t, :, :])
		#print(utility_zeros.shape)
		output_matrix[t, :, :, 0] += channel2_array_normed[t, :, :]*(1 - mixing_ratio)*color2[0]*(foreground_prob[t,:,:] > bg_thresh)
		output_matrix[t, :, :, 1] += channel2_array_normed[t, :, :]*(1 - mixing_ratio)*color2[1]*(foreground_prob[t,:,:] > bg_thresh)
		output_matrix[t, :, :, 2] += channel2_array_normed[t, :, :]*(1 - mixing_ratio)*color2[2]*(foreground_prob[t,:,:] > bg_thresh)
	
	FFMpegWriter = manimation.writers['ffmpeg']
	metadata = dict(title=outfile_name.strip('.mp4'), artist='Matplotlib', comment='Movie support!')
	writer = FFMpegWriter(fps=15, metadata=metadata)
	
	figdimx = int(nx/200)
	figdimy = int(ny/200)
	
	FFMpegWriter = manimation.writers['ffmpeg']
	metadata = dict(title=outfile_name.strip('.mp4'), artist='Matplotlib', comment='Movie support!')
	writer = FFMpegWriter(fps=15, metadata=metadata)
	
	output_matrix_xyswap = numpy.swapaxes( output_matrix, 1, 2 )
	
	fig = pt.figure(figsize=(figdimx,figdimy))
	l = pt.imshow(output_matrix_xyswap[0,:,:,:]/numpy.max(output_matrix[0,:,:,:]))
	
	with writer.saving(fig, outfile_name, dpi=200):
		for t in range(min(ntsteps,maxtsteps)):
			
			l.set_data( output_matrix_xyswap[t,:,:,:]/numpy.max(output_matrix[t,:,:,:]) )
			writer.grab_frame()
	
	return output_matrix

if __name__=='__main__':
	
	import sys
	
	bf_image = sys.argv[1]
	fluor_stack = sys.argv[2]
	output_filename = sys.argv[3]
	lower_thresh = float(sys.argv[4])
	upper_thresh = float(sys.argv[5])
	max_tsteps = int(sys.argv[6])
	
	make_overlay_movie( bf_image, fluor_stack, output_filename, lower_threshold=lower_thresh, upper_threshold=upper_thresh, maxtsteps=max_tsteps )
	
	