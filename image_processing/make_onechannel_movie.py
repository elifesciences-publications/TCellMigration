def make_onechannel_movie( image_data_file, outfile_name, lower_threshold = 0 ):
	
	###This function takes a .tif stack as input and makes an .mp4 movie.
	
	###Parameters:
	
	###image_data: numpy array containing images; TxYxX
	###outfile_name: output file to write to; should be .mp4
	###lower_threshold: pixel values will be thresholded frame-by-frame at the percentile chosen as 'lower_threshold'. Default is 0.
	
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pylab as pt
	import matplotlib.animation as manimation
	import numpy
	from skimage import io
	
	image_data = io.imread( image_data_file )
	
	ntsteps, nx, ny = image_data.shape
	
	figdimx = int(nx/200)
	figdimy = int(ny/200)
	
	FFMpegWriter = manimation.writers['ffmpeg']
	metadata = dict(title=outfile_name.strip('.mp4'), artist='Matplotlib', comment='Movie support!')
	writer = FFMpegWriter(fps=15, metadata=metadata)
	
	pt.gray()
	
	fig = pt.figure(figsize=(figdimx,figdimy))
	l = pt.imshow(image_data[0,:,:].T/numpy.max(image_data[0,:,:]))
	ax = pt.gca()
	with writer.saving(fig, outfile_name, dpi=400):
		for t in range(ntsteps):
			
			#drift_mask = numpy.nonzero(image_data[t, :, :]) ###
		
			frame_min = numpy.percentile(image_data[t, :, :], lower_threshold)
			
			frame_max = numpy.max(image_data[t, :, :])
			
			l.set_data( (image_data[t,:,:].T - frame_min)/(frame_max - frame_min))
			ax.set_yticks([])
			ax.set_xticks([])
			
			writer.grab_frame()

if __name__=="__main__":
	
	import sys
	
	image_data = sys.argv[1]
	output_file = sys.argv[2]
	
	lower_threshold = float(sys.argv[3])
	
	make_onechannel_movie(image_data, output_file, lower_threshold)