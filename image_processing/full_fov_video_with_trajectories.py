def make_cell_movie(timeseries,trajectory_filename,merger_filename,output_base,timestep,ntmax):

	####This function makes an .mp4 video overlaying a MaxZ stack with trajectories.
	####Inputs:
	####timeseries: maxz projection in .tif format
	####trajectory_filename: trajectories.
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
	
	
	mergers_by_index = {}
	
	merger_file = open(merger_filename,'r')
	firstline=True
	for line in merger_file:
		if firstline:
			firstline = False
		else:
			linelist=line.strip().split(',')
			if len(linelist) > 1.5:
				t = int(linelist[0])
				indexes = [int(s) for s in linelist[2].split(';')]
				for index in indexes:
					if index not in mergers_by_index:
						mergers_by_index[index] = {}
					mergers_by_index[index][t] = [int(ind) for ind in indexes] ###All tracks collided at this timestep, including the focal track
	merger_file.close()
	
	traj_file = open(trajectory_filename,'r')
	track_dict = {}
	for line in traj_file:
		linelist = line.strip().split('\t')
		index = int(linelist[0])
		track_dict[index] = {}
		track_dict[index]['tlist'] = []
		track_dict[index]['coords'] = []
		for txyz in linelist[1:]:
			t,x,y,z = txyz.split(',')
			track_dict[index]['tlist'].append(int(t))
			track_dict[index]['coords'].append([float(x),float(y)])
	traj_file.close()
	
	#speed_edges = numpy.array([0,2,7,11,14])
	chosen_indexes = {15,24,11,30}
	#chosen_indexes = {15,20,11,5}
	color_cycle = {24:'goldenrod',15:'C6',30:'C9',11:'C5'}
	#colors = {0:'C2',1:'C1',2:'C0',3:'C4',4:'C5'}
	#color_cycle = {}
	
	outfile_name = output_base + '_trajs_lowres.mp4'
	image_data = io.imread(timeseries)
	ntsteps,nx,ny = image_data.shape
	
	ntsteps = min(ntsteps,ntmax)
	
	figdimy = nx/100
	figdimx = ny/100
	
	####Save the final frame in .pdf format, with the colormap reversed for the microscopy image
	fig = pt.figure()
	fig.set_size_inches((figdimx,figdimy),forward=True)
	ax = fig.subplots(1,1)
	fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
	l = ax.imshow(image_data[0,:,:]/numpy.maximum(numpy.max(image_data[0,:,:]),1),cmap='gray')
	ax.xaxis.set_visible(False)
	ax.yaxis.set_visible(False)
	ax.set_frame_on(False)
	scale_bar_start_x = 50
	scale_bar_start_y = 100
	# set figure background opacity (alpha) to 0
	fig.patch.set_alpha(0.)
	for t in range(ntsteps)[1:]:
		#pt.cla()
		for index in track_dict:
			if index not in chosen_indexes:
				traj = numpy.array(track_dict[index]['coords'])
				tlist = track_dict[index]['tlist']
				
				if t in tlist:
					tind = tlist.index(t)
					if tind > .5 and (0 < traj[tind-1,1] < ny and 0< traj[tind,1] < ny and 0 < traj[tind-1,0]< nx and 0 < traj[tind,0] < nx): ###Need to not erroneously plot the final tp connected to the first tp for trajectories that do not start at frame 0
						if 0 < traj[tind-1,1] < ny and 0 < traj[tind,1] < ny and 0 < traj[tind-1,0] < nx and 0 < traj[tind,0] < nx: ###Due to drift subtraction, it is possible for a trajectory to go slightly out of the current frame; do not plot this
							ax.plot( [traj[tind-1,1],traj[tind,1]], [traj[tind-1,0],traj[tind,0]], linewidth=.5, color = 'Gray', alpha=1)
			else:
				traj = numpy.array(track_dict[index]['coords'])
				tlist = track_dict[index]['tlist']
				
				if t in tlist:
					tind = tlist.index(t)
					if tind > .5 and (0 < traj[tind-1,1] < ny and 0< traj[tind,1] < ny and 0 < traj[tind-1,0]< nx and 0 < traj[tind,0] < nx): ###Need to not erroneously plot the final tp connected to the first tp for trajectories that do not start at frame 0
						ax.plot( [traj[tind-1,1],traj[tind,1]], [traj[tind-1,0],traj[tind,0]], linewidth=2, color = color_cycle[index])
	####Add a scale bar
			
	ax.plot([scale_bar_start_x,scale_bar_start_x + 4*33.7],[scale_bar_start_y,scale_bar_start_y], color = 'w', linewidth = 6)
	ax.text(scale_bar_start_x + 2*33.7, scale_bar_start_y - 40, "100 $\mu m$", horizontalalignment='center', verticalalignment='center', color='w', fontsize=36)
	pt.savefig(output_base + '_trajs_still4.pdf',bbox_inches='tight')
	pt.close()
		
	
	FFMpegWriter = manimation.writers['ffmpeg']
	metadata = dict(title=outfile_name.strip('.mp4'), artist='Matplotlib', comment='Movie support!')
	writer = FFMpegWriter(fps=20, metadata=metadata)
	

	fig = pt.figure()
	fig.set_size_inches((figdimx,figdimy),forward=True)
	ax = fig.subplots(1,1)
	fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
		
	with writer.saving(fig, outfile_name, dpi=100):
		scale_bar_start_x = 50
		scale_bar_start_y = 100
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
			for index in track_dict:
				if index not in chosen_indexes:
					traj = numpy.array(track_dict[index]['coords'])
					tlist = track_dict[index]['tlist']
					
					if t in tlist:
						tind = tlist.index(t)
						if tind > .5 and (0 < traj[tind-1,1] < ny and 0< traj[tind,1] < ny and 0 < traj[tind-1,0]< nx and 0 < traj[tind,0] < nx): ###Need to not erroneously plot the final tp connected to the first tp for trajectories that do not start at frame 0
							if 0 < traj[tind-1,1] < ny-1 and 0 < traj[tind,1] < ny-1 and 0 < traj[tind-1,0] < nx-1 and 0 < traj[tind,0] < nx-1: ###Due to drift subtraction, it is possible for a trajectory to go slightly out of the current frame; do not plot this
								ax.plot( [traj[tind-1,1],traj[tind,1]], [traj[tind-1,0],traj[tind,0]], linewidth=.5, color = 'Gray', alpha=.8)
				else:
					traj = numpy.array(track_dict[index]['coords'])
					tlist = track_dict[index]['tlist']
					
					if t in tlist:
						tind = tlist.index(t)
						if tind > .5 and (0 < traj[tind-1,1] < ny and 0< traj[tind,1] < ny and 0 < traj[tind-1,0]< nx and 0 < traj[tind,0] < nx): ###Need to not erroneously plot the final tp connected to the first tp for trajectories that do not start at frame 0
							ax.plot( [traj[tind-1,1],traj[tind,1]], [traj[tind-1,0],traj[tind,0]], linewidth=.9, color = color_cycle[index])
			
			####Add a scale bar and a timestamp
			
			ax.plot([scale_bar_start_x,scale_bar_start_x + 2*33.7],[scale_bar_start_y,scale_bar_start_y], color = 'w', linewidth = 3)
			ax.text(scale_bar_start_x + 33.7, scale_bar_start_y - 15, "50 $\mu m$", horizontalalignment='center', verticalalignment='center', color='w', fontsize=14)
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
			ts = ax.text(scale_bar_start_x + 33.7, scale_bar_start_y - 50, timestamp, horizontalalignment='center', verticalalignment='center', color='w', fontsize=14)
			
			writer.grab_frame()
			
			
			
			if t%100 == 0:
				print('timestep ',t)
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
	make_cell_movie(timeseries,mask_filename,track_file,merger_file,output_base)