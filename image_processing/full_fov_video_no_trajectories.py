def make_cell_movie(timeseries,output_base,timestep,ntmax):

	####This function makes an .mp4 video with a timestamp.
	####Inputs:
	####timeseries: maxz projection in .tif format
	####output_base: path and base name for saving movie
	
	import matplotlib
	#matplotlib.use("Agg")
	import matplotlib.pylab as pt
	import matplotlib.animation as manimation
	#from glob import glob
	from skimage import io, measure
	#from scipy.ndimage import measurements
	#from scipy.stats import moment
	import h5py
	import numpy
	from matplotlib import cm
	from matplotlib.colors import ListedColormap, LinearSegmentedColormap
	
	
	###Choose 3 cells to highlight
	# chosen_indexes = set()
	# speed_grps = set()
	# for index in track_dict:
		# tlist = numpy.array(track_dict[index]['tlist'])
		# tdiffs = tlist[1:] - tlist[:-1]
		# coords = numpy.array(track_dict[index]['coords'])
		# speeds = numpy.sqrt(numpy.sum((coords[1:,:] - coords[:-1,:])**2,axis=1))/tdiffs
		# mean_speed = numpy.mean(speeds)*60/timestep
		
		# speed_grp = numpy.argmin(speed_edges - mean_speed < 0)
		# print(speed_grp)
		# if tlist[0] < .5 and len(tlist) > 800 and speed_grp not in speed_grps:
			# chosen_indexes.add(index)
			# speed_grps.add(speed_grp)
			# color_cycle[index] = colors[speed_grp]
	# print(chosen_indexes)		
	outfile_name = output_base + '_timeseries_lowres.mp4'
	image_data = io.imread(timeseries)
	
	ntsteps,nx,ny = image_data.shape
	
	ntsteps = min(ntsteps,ntmax)
	
	FFMpegWriter = manimation.writers['ffmpeg']
	metadata = dict(title=outfile_name.strip('.mp4'), artist='Matplotlib', comment='Movie support!')
	writer = FFMpegWriter(fps=15, metadata=metadata)
	
	figdimy = nx/100
	figdimx = ny/100
		
	fig = pt.figure()
	fig.set_size_inches((figdimx,figdimy),forward=True)
	ax = fig.subplots(1,1)
	fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
		
	with writer.saving(fig, outfile_name, dpi=50):
		scale_bar_start_x = 200
		scale_bar_start_y = 240
		l = ax.imshow(image_data[0,:,:]/numpy.maximum(numpy.max(image_data[0,:,:]),1),cmap='gray')
		# turn off axis spines
		ax.xaxis.set_visible(False)
		ax.yaxis.set_visible(False)
		ax.set_frame_on(False)
				
		# set figure background opacity (alpha) to 0
		fig.patch.set_alpha(0.)
		for t in range(ntsteps)[1:]:
			#pt.cla()
			l.set_data( image_data[t,:,:]/numpy.maximum(numpy.max(image_data[t,:,:]),1) )
			
			
			####Add a (scale bar) and a timestamp
			
			ax.plot([scale_bar_start_x,scale_bar_start_x + 100/.37],[scale_bar_start_y,scale_bar_start_y], color = 'w', linewidth = 6)
			ax.text(scale_bar_start_x + 50/.37, scale_bar_start_y - 50, "100 $\mu m$", horizontalalignment='center', verticalalignment='center', color='w', fontsize=50)
			total_secs = t*timestep
			
			hrs = int(numpy.floor(total_secs/3600))
			mins = int(numpy.floor((total_secs % 3600)/60))
			secs = total_secs - hrs*3600 - mins*60
			
			hr_str = str(hrs)
			min_str = str(mins).zfill(2)
			sec_str = str(secs).zfill(2)
			timestamp = hr_str + ':' + min_str + ':' + sec_str
			if t > 1.5:
				ts.remove()
			ts = ax.text(scale_bar_start_x + 50/.37, scale_bar_start_y - 160, timestamp, horizontalalignment='center', verticalalignment='center', color='w', fontsize=50)
			
			writer.grab_frame()
			
			####Save the final frame in .pdf format
			
			#if t%100 == 0:
			#	print('timestep ',t)
			#if t > ntsteps - 1.5:
				
				#pt.savefig(output_base + '_trajs_still2.pdf',bbox_inches='tight')
				
		pt.close()
if __name__=="__main__":
	
	import sys
	timeseries = sys.argv[1]
	mask_filename = sys.argv[2]
	track_file = sys.argv[3]
	merger_file = sys.argv[4]
	output_base	= sys.argv[5]
	make_cell_movie(timeseries,output_base,timestep=100,ntmax=1000)