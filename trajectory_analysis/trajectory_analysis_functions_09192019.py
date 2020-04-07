import numpy

def import_data_merged_file( trajectory_filename ):
	
	master_trajectory_dict = {}
	
	trajectory_file = open(trajectory_filename,'r')
	
	header_line = True
	for line in trajectory_file:
		if header_line:
			header_line = False
			continue
		linelist = line.strip().split('\t')
		experiment_type = linelist[0]
		treatment = linelist[1]
		sample = linelist[2]
		cell_index = linelist[3]
		
		if experiment_type not in master_trajectory_dict:
			master_trajectory_dict[experiment_type] = {}
		if treatment not in master_trajectory_dict[experiment_type]:
			master_trajectory_dict[experiment_type][treatment] = {}
		if sample not in master_trajectory_dict[experiment_type][treatment]:
			master_trajectory_dict[experiment_type][treatment][sample] = {}
		
		master_trajectory_dict[experiment_type][treatment][sample][cell_index] = []
		
		for txy in linelist[4].split(';'):
			
			try:
				t,x,y = [float(l) for l in txy.split(',')]
				if ~numpy.isnan(x) and ~numpy.isnan(y):
					master_trajectory_dict[experiment_type][treatment][sample][cell_index].append((t,x,y))
			except:
				print("Non-numerical trajectory entry:", experiment_type,treatment,sample,cell_index,txy)
				
	trajectory_file.close()
	
	return master_trajectory_dict			


def interpolate_timepoints( trajectory_dict_polar, nmissing = 1 ):
	
	####This function takes trajectories and interpolates missing timepoints.
	####If more than nmissing consecutive timepoint is missing, the trajectory is truncated either before or after the gap, taking the longer consecutive piece. (Default: 1)
	####To truncate each trajectory to the longest segment with no missing timesteps, use nmissing = 0.
	
	trajectory_dict_polar_interpolated = {}
	for experiment in trajectory_dict_polar:
		trajectory_dict_polar_interpolated[experiment] = {}
		for treatment in trajectory_dict_polar[experiment]:
			trajectory_dict_polar_interpolated[experiment][treatment] = {}
			for sample in trajectory_dict_polar[experiment][treatment]:
				trajectory_dict_polar_interpolated[experiment][treatment][sample] = {}
				for traj_ind in trajectory_dict_polar[experiment][treatment][sample]:
					traj_data = trajectory_dict_polar[experiment][treatment][sample][traj_ind]
					interval = numpy.min(traj_data[3]) ###Minimum interval
					
					n_timesteps_anticipated = (traj_data[0][-1] - traj_data[0][0])/interval + 1
					n_timesteps_real = len(traj_data[0])
					
					if n_timesteps_real + .5 >= n_timesteps_anticipated:
						trajectory_dict_polar_interpolated[experiment][treatment][sample][traj_ind] = traj_data ###No missing timesteps
					else:
						temporary_traj_dict = {}
						traj_section_ind = 0
						traj_section_lengths = []
						
						for tp in range(n_timesteps_real):
							if traj_data[3][tp] - interval < interval/2.: ###Not a missing tp
								if traj_section_ind not in temporary_traj_dict:
									temporary_traj_dict[traj_section_ind] = [[],[],[],[]]
								for k in range(4):
									temporary_traj_dict[traj_section_ind][k].append( traj_data[k][tp] )
							elif traj_data[3][tp] - (nmissing+1)*interval < interval/2.:
								n_tps = int(traj_data[3][tp]/interval)
								if traj_section_ind not in temporary_traj_dict:
									temporary_traj_dict[traj_section_ind] = [[],[],[],[]]
								for seg_tps in range(n_tps):
									temporary_traj_dict[traj_section_ind][0].append( traj_data[0][tp] + seg_tps*interval ) ###Timepoint (including the first 'real' one)
									temporary_traj_dict[traj_section_ind][1].append( traj_data[1][tp]/float(n_tps) ) ####Segment length divided by the number of timesteps
									temporary_traj_dict[traj_section_ind][2].append( traj_data[2][tp] ) ###Angle is the same for all interpolated steps
									temporary_traj_dict[traj_section_ind][3].append( interval )
								
							elif traj_data[3][tp] - (nmissing+1)*interval > interval/2.:
								###Truncate
								if traj_section_ind not in temporary_traj_dict:
									traj_section_lengths.append(0)
								else:
									traj_section_lengths.append(len(temporary_traj_dict[traj_section_ind][0]))
								traj_section_ind += 1
						###append length of final segment		
						if traj_section_ind in temporary_traj_dict:
							traj_section_lengths.append(len(temporary_traj_dict[traj_section_ind][0]))	
						
						longest_segment = numpy.argmax(traj_section_lengths)
					
						trajectory_dict_polar_interpolated[experiment][treatment][sample][traj_ind] = temporary_traj_dict[longest_segment]
						
	return trajectory_dict_polar_interpolated
	
def trajectories_polar( master_trajectory_dict ):
	
	###Return displacements that make up each trajectory in the format [initial_timepoints,xy_lengths,xy_angles,delta_ts]
	###Note that the angles specify the direction of the displacement vector in an absolute coordinate system, and are defined on the interval [-pi,pi]
	trajectories_polar_dict = {}
	for experiment in master_trajectory_dict:
		trajectories_polar_dict[experiment] = {}
		for treatment in master_trajectory_dict[experiment]:
			trajectories_polar_dict[experiment][treatment] = {}
			for sample in master_trajectory_dict[experiment][treatment]:
				trajectories_polar_dict[experiment][treatment][sample] = {}
				for traj_ind in master_trajectory_dict[experiment][treatment][sample]:
					
					trajectory = numpy.array(master_trajectory_dict[experiment][treatment][sample][traj_ind])
					
					if trajectory.shape[0] > 9: ###Omit all trajectories with fewer than 10 timepoints
						delta_trajectory = trajectory[1:,:] - trajectory[:-1,:]
						
						rs = numpy.sqrt( delta_trajectory[:,1]**2 + delta_trajectory[:,2]**2 )
						
						thetas = numpy.sign(delta_trajectory[:,2])*numpy.arccos(delta_trajectory[:,1]/rs)
						
						thetas[rs == 0] = 0
						if numpy.min(rs) <= 0:
							thetas[rs == 0] = 0
							#print('zero displacement?',experiment,treatment,sample)
							#print(trajectory[numpy.argmin(rs),:],trajectory[numpy.argmin(rs)+1,:])
							
						delta_ts = delta_trajectory[:,0]
						if numpy.sum(numpy.isnan(delta_ts)) +  numpy.sum(numpy.isnan(rs)) > .5:
							print('nan???',experiment,treatment,sample,traj_ind)
						trajectories_polar_dict[experiment][treatment][sample][traj_ind] = [trajectory[:-1,0], rs, thetas, delta_ts]
	
	return trajectories_polar_dict

def calculate_corr_times_one_trajectory( time_intervals, thetas, overlap ):

	###Helper function for correlation_times_by_cell, for a portion of a trajectory
	
	i = 0
	theta0 = thetas[0]
	corr_times = []
	corr_time = 0
	if not overlap:
		while i < len(thetas)-1.5:
			
			if numpy.abs(thetas[i+1] - theta0) < numpy.pi/2.:
				corr_time += time_intervals[i]
				
			else:
				
				corr_times.append(corr_time)
				theta0 = thetas[i+1]
				corr_time = 0
				
			i += 1
	else:
		base_ind = 0
		while i < len(thetas)-1.5:
			
			if numpy.abs(thetas[i+1] - theta0) < numpy.pi/2.:
				corr_time += time_intervals[i]
				
			else:
				
				corr_times.append(corr_time)
				base_ind += 1
				theta0 = thetas[base_ind]
				corr_time = 0
				i = base_ind - 1
				
			i += 1
	###Remove the final (potentially truncated) segment
	if len(corr_times) > .5:
		corr_times = corr_times[:-1]
	return corr_times

def calculate_corr_times_one_trajectory_shallow( time_intervals, thetas, overlap ):

	###Helper function for correlation_times_by_cell, for a portion of a trajectory
	
	i = 0
	theta0 = thetas[0]
	corr_times = []
	corr_time = 0
	if not overlap:
		while i < len(thetas)-1.5:
			
			if numpy.abs(thetas[i+1] - theta0) < numpy.pi/6.:
				corr_time += time_intervals[i]
				
			else:
				
				corr_times.append(corr_time)
				theta0 = thetas[i+1]
				corr_time = 0
				
			i += 1
	else:
		base_ind = 0
		while i < len(thetas)-1.5:
			
			if numpy.abs(thetas[i+1] - theta0) < numpy.pi/6.:
				corr_time += time_intervals[i]
				
			else:
				
				corr_times.append(corr_time)
				base_ind += 1
				theta0 = thetas[base_ind]
				corr_time = 0
				i = base_ind - 1
				
			i += 1
	###Remove the final (potentially truncated) segment
	if len(corr_times) > .5:
		corr_times = corr_times[:-1]
	return corr_times

def calculate_corr_times_with_speeds( rs, time_intervals, thetas, overlap ):
	###Helper function for correlation_times_by_cell, for a portion of a trajectory
	
	i = 0
	theta0 = thetas[0]
	corr_times = []
	r0 = rs[0]
	rlist = []
	corr_time = 0
	if not overlap:
		
		while i < len(thetas)-1.5:
			
			if numpy.abs(thetas[i+1] - theta0) < numpy.pi/2.:
				corr_time += time_intervals[i]
				
			else:
				
				corr_times.append(corr_time)
				rlist.append(r0)
				theta0 = thetas[i+1]
				r0 = rs[i+1]
				corr_time = 0
				
			i += 1
	else:
		base_ind = 0
		while i < len(thetas)-1.5:
			
			if numpy.abs(thetas[i+1] - theta0) < numpy.pi/2.:
				corr_time += time_intervals[i]
				
			else:
				
				corr_times.append(corr_time)
				rlist.append(r0)
				base_ind += 1
				theta0 = thetas[base_ind]
				r0 = rs[base_ind]
				corr_time = 0
				i = base_ind - 1
				
			i += 1
	###Remove the final (potentially truncated) segment
	if len(corr_times) > .5:
		corr_times = corr_times[:-1]
		rlist = rlist[:-1]
	return numpy.array([rlist,corr_times])

def calculate_corr_times_with_speeds_shallow( rs, time_intervals, thetas, overlap ):
	###Helper function for correlation_times_by_cell, for a portion of a trajectory
	
	i = 0
	theta0 = thetas[0]
	corr_times = []
	r0 = rs[0]
	rlist = []
	corr_time = 0
	if not overlap:
		
		while i < len(thetas)-1.5:
			
			if numpy.abs(thetas[i+1] - theta0) < numpy.pi/6.:
				corr_time += time_intervals[i]
				
			else:
				
				corr_times.append(corr_time)
				rlist.append(r0)
				theta0 = thetas[i+1]
				r0 = rs[i+1]
				corr_time = 0
				
			i += 1
	else:
		base_ind = 0
		while i < len(thetas)-1.5:
			
			if numpy.abs(thetas[i+1] - theta0) < numpy.pi/6.:
				corr_time += time_intervals[i]
				
			else:
				
				corr_times.append(corr_time)
				rlist.append(r0)
				base_ind += 1
				theta0 = thetas[base_ind]
				r0 = rs[base_ind]
				corr_time = 0
				i = base_ind - 1
				
			i += 1
	###Remove the final (potentially truncated) segment
	if len(corr_times) > .5:
		corr_times = corr_times[:-1]
		rlist = rlist[:-1]
	return numpy.array([rlist,corr_times])
	
def calculate_corr_lengths_one_trajectory( rs, thetas, overlap ):

	###Helper function for correlation_times_by_cell, for a portion of a trajectory
	
	i = 0
	theta0 = thetas[0]
	r0 = rs[0]
	corr_times = []
	corr_time = rs[0]
	if not overlap:
		while i < len(thetas)-1.5:
			
			if numpy.abs(thetas[i+1] - theta0) < numpy.pi/2.:
				corr_time += rs[i+1]*numpy.cos(thetas[i+1])*numpy.cos(theta0) + rs[i+1]*numpy.sin(thetas[i+1])*numpy.sin(theta0)
				
			else:
				
				corr_times.append(corr_time)
				theta0 = thetas[i+1]
				r0 = rs[i+1]
				corr_time = r0
				
			i += 1
	else:
		base_ind = 0
		while i < len(thetas)-1.5:
			
			if numpy.abs(thetas[i+1] - theta0) < numpy.pi/2.:
				corr_time += rs[i+1]*numpy.cos(thetas[i+1])*numpy.cos(theta0) + rs[i+1]*numpy.sin(thetas[i+1])*numpy.sin(theta0)
				
			else:
				
				corr_times.append(corr_time)
				base_ind += 1
				theta0 = thetas[base_ind]
				r0 = rs[base_ind]
				corr_time = r0
				i = base_ind - 1
				
			i += 1
	###Remove the final (potentially truncated) segment
	if len(corr_times) > .5:
		corr_times = corr_times[:-1]
	return corr_times

def extract_onestep_couplings( rs, thetas ):
	
	###Helper function for length_angle_couplings_by_cell; returns a list of ((r1,r2),theta) for all internal angles
	
	coupling_list = []
	for t in range(len(rs)-1):
		
		r0 = rs[t]
		r1 = rs[t+1]
		delta_theta = numpy.abs(thetas[t+1] - thetas[t])
		if delta_theta > numpy.pi:
			delta_theta = 2*numpy.pi - delta_theta
		
		coupling_list.append( ((r0,r1),delta_theta) )
	
	return coupling_list
	
def bout_lengthsx_one_trajectory( trajectory_polar, overlap = False ):

	x_lens = trajectory_polar[:,1]*numpy.cos(trajectory_polar[:,2])
	###Find sign switches and distances between them
	
	xdirs = numpy.sign(x_lens)
	xdir_diffs = xdirs[1:] - xdirs[:-1]
	bout_lens = []
	
	i = 0
	while i < len(xdirs)-1:
		bout_len = x_lens[i]
		j = i+1
		while j<len(xdir_diffs) and numpy.abs(xdir_diffs[j-1]) < .5:
			bout_len += x_lens[j]
			j += 1
		if j<len(xdirs):
			bout_lens.append(numpy.abs(bout_len))
		i = j
		
	return bout_lens

def bout_lengthsx( trajectory_dict_polar_interp ):
	
	bout_length_dict = {}
	for experiment in trajectory_dict_polar_interp:
		bout_length_dict[experiment] = {}
		for treatment in trajectory_dict_polar_interp[experiment]:
			bout_length_dict[experiment][treatment] = {}
			for sample in trajectory_dict_polar_interp[experiment][treatment]:
				bout_length_dict[experiment][treatment][sample] = {}
				for traj_ind in trajectory_dict_polar_interp[experiment][treatment][sample]:
					traj = numpy.array(trajectory_dict_polar_interp[experiment][treatment][sample][traj_ind]).T
					bout_lens = bout_lengthsx_one_trajectory( traj, overlap = False )
					bout_length_dict[experiment][treatment][sample][traj_ind] = bout_lens
	
	return bout_length_dict
	
def correlation_times_lengths_by_cell( trajectories_polar, calculation='times', overlap=False ):
	
	###Calculate the correlation time or correlation length in non-overlapping intervals along each trajectory. For correlation length rather than time, use calculation='lengths'
	
	correlation_times_by_cell = {}
	speeds_by_cell = {}
	
	for experiment in trajectories_polar:
		correlation_times_by_cell[experiment] = {}
		speeds_by_cell[experiment] = {}
		for treatment in trajectories_polar[experiment]:
			correlation_times_by_cell[experiment][treatment] = {}
			speeds_by_cell[experiment][treatment] = {}
			for sample in trajectories_polar[experiment][treatment]:
				correlation_times_by_cell[experiment][treatment][sample] = {}
				speeds_by_cell[experiment][treatment][sample] = {}
				first_traj = True
				for traj_ind in trajectories_polar[experiment][treatment][sample]:
				
					polar_traj = trajectories_polar[experiment][treatment][sample][traj_ind]
					dts = polar_traj[3]
					if min(dts) <= 0:
						print(treatment,experiment,sample)
						print(dts)
					thetas = polar_traj[2]
					rs = polar_traj[1]
					
					if first_traj:
						interval = numpy.min(dts)
						print(interval)
						if interval <= 0:
							print(dts)
							print('Times not monotonic!',experiment,treatment,sample,traj_ind)
							raise ValueError
						first_traj = False
					
					dt_intervals = (dts/interval).astype('int')
					#print(dt_intervals)	
					####Find ranges within each trajectory where there are no missing timesteps
					
					skips = numpy.arange(len(dt_intervals))[dt_intervals > 1.5] ###Find all segments associated with one or more missing timepoints
					consecutive_segs = []
					initial_ind = 0
					for skip in skips:
						if skip - initial_ind > 2.5:
							
							final_ind = skip ###This is the final slice index to be used for the consecutive block
							consecutive_segs.append([initial_ind,final_ind])
						initial_ind = skip + 1
					if len(skips) > .5 and initial_ind < len(dt_intervals)-1:
						consecutive_segs.append([initial_ind,len(dt_intervals)])
					if len(skips) < .5:
						consecutive_segs.append([0,len(dt_intervals)])
						
					rs_cons = []
					dts_cons = []
					
					for seg in consecutive_segs:
						rs_cons.extend( rs[seg[0]:seg[1]] )
						dts_cons.extend( dts[seg[0]:seg[1]] )
						if calculation == 'times':
							corr_times = calculate_corr_times_one_trajectory( dts[seg[0]:seg[1]], thetas[seg[0]:seg[1]], overlap )
						elif calculation == 'lengths':
							corr_times = calculate_corr_lengths_one_trajectory( rs[seg[0]:seg[1]], thetas[seg[0]:seg[1]], overlap )
					
						if len(corr_times) > .5: ###There is at least one non-truncated bout in this trajectory segment
							if traj_ind not in correlation_times_by_cell[experiment][treatment][sample]:
								correlation_times_by_cell[experiment][treatment][sample][traj_ind] = []
							correlation_times_by_cell[experiment][treatment][sample][traj_ind].extend(corr_times)
					#speeds_by_cell[experiment][treatment][sample][traj_ind] = rs[dt_intervals<1.5]/dts[dt_intervals<1.5]*60		
					
					speeds_by_cell[experiment][treatment][sample][traj_ind] = numpy.array(rs_cons)/numpy.array(dts_cons)*60 ###Speeds in microns/min, measured on consecutive steps (secant approx)
				
	return correlation_times_by_cell, speeds_by_cell

def correlation_times_lengths_by_cell_shallow( trajectories_polar, calculation='times', overlap=False ):
	
	###Calculate the correlation time or correlation length in non-overlapping intervals along each trajectory. For correlation length rather than time, use calculation='lengths'
	
	correlation_times_by_cell = {}
	speeds_by_cell = {}
	
	for experiment in trajectories_polar:
		correlation_times_by_cell[experiment] = {}
		speeds_by_cell[experiment] = {}
		for treatment in trajectories_polar[experiment]:
			correlation_times_by_cell[experiment][treatment] = {}
			speeds_by_cell[experiment][treatment] = {}
			for sample in trajectories_polar[experiment][treatment]:
				correlation_times_by_cell[experiment][treatment][sample] = {}
				speeds_by_cell[experiment][treatment][sample] = {}
				first_traj = True
				for traj_ind in trajectories_polar[experiment][treatment][sample]:
				
					polar_traj = trajectories_polar[experiment][treatment][sample][traj_ind]
					dts = polar_traj[3]
					if min(dts) <= 0:
						print(treatment,experiment,sample)
						print(dts)
					thetas = polar_traj[2]
					rs = polar_traj[1]
					
					if first_traj:
						interval = numpy.min(dts)
						print(interval)
						if interval <= 0:
							print(dts)
							print('Times not monotonic!',experiment,treatment,sample,traj_ind)
							raise ValueError
						first_traj = False
					
					dt_intervals = (dts/interval).astype('int')
					#print(dt_intervals)	
					####Find ranges within each trajectory where there are no missing timesteps
					
					skips = numpy.arange(len(dt_intervals))[dt_intervals > 1.5] ###Find all segments associated with one or more missing timepoints
					consecutive_segs = []
					initial_ind = 0
					for skip in skips:
						if skip - initial_ind > 2.5:
							
							final_ind = skip ###This is the final slice index to be used for the consecutive block
							consecutive_segs.append([initial_ind,final_ind])
						initial_ind = skip + 1
					if len(skips) > .5 and initial_ind < len(dt_intervals)-1:
						consecutive_segs.append([initial_ind,len(dt_intervals)])
					if len(skips) < .5:
						consecutive_segs.append([0,len(dt_intervals)])
						
					rs_cons = []
					dts_cons = []
					
					for seg in consecutive_segs:
						rs_cons.extend( rs[seg[0]:seg[1]] )
						dts_cons.extend( dts[seg[0]:seg[1]] )
						if calculation == 'times':
							corr_times = calculate_corr_times_one_trajectory_shallow( dts[seg[0]:seg[1]], thetas[seg[0]:seg[1]], overlap )
						elif calculation == 'lengths':
							corr_times = calculate_corr_lengths_one_trajectory_shallow( rs[seg[0]:seg[1]], thetas[seg[0]:seg[1]], overlap )
					
						if len(corr_times) > .5: ###There is at least one non-truncated bout in this trajectory segment
							if traj_ind not in correlation_times_by_cell[experiment][treatment][sample]:
								correlation_times_by_cell[experiment][treatment][sample][traj_ind] = []
							correlation_times_by_cell[experiment][treatment][sample][traj_ind].extend(corr_times)
					#speeds_by_cell[experiment][treatment][sample][traj_ind] = rs[dt_intervals<1.5]/dts[dt_intervals<1.5]*60		
					
					speeds_by_cell[experiment][treatment][sample][traj_ind] = numpy.array(rs_cons)/numpy.array(dts_cons)*60 ###Speeds in microns/min, measured on consecutive steps (secant approx)
				
	return correlation_times_by_cell, speeds_by_cell

def length_angle_couplings_by_cell( trajectories_polar ):
	
	###For all pieces of each trajectory with all adjacent steps, collect lists of ((r1,r2),delta_theta) between adjacent steps
	
	couplings_by_cell = {}
	speeds_by_cell = {}
	
	for experiment in trajectories_polar:
		couplings_by_cell[experiment] = {}
		speeds_by_cell[experiment] = {}
		for treatment in trajectories_polar[experiment]:
			couplings_by_cell[experiment][treatment] = {}
			speeds_by_cell[experiment][treatment] = {}
			for sample in trajectories_polar[experiment][treatment]:
				couplings_by_cell[experiment][treatment][sample] = {}
				speeds_by_cell[experiment][treatment][sample] = {}
				first_traj = True
				for traj_ind in trajectories_polar[experiment][treatment][sample]:
				
					polar_traj = trajectories_polar[experiment][treatment][sample][traj_ind]
					dts = polar_traj[3]
					if min(dts) <= 0:
						print(treatment,experiment,sample)
						print(dts)
					thetas = polar_traj[2]
					rs = polar_traj[1]
					
					if first_traj:
						interval = numpy.min(dts)
						print(interval)
						if interval <= 0:
							print(dts)
							print('Times not monotonic!',experiment,treatment,sample,traj_ind)
							raise ValueError
						first_traj = False
					
					dt_intervals = (dts/interval).astype('int')
						
					####Find ranges within each trajectory where there are no missing timesteps
					
					skips = numpy.arange(len(dt_intervals))[dt_intervals > 1.5] ###Find all segments associated with one or more missing timepoints
					consecutive_segs = []
					initial_ind = 0
					for skip in skips:
						if skip - initial_ind > 2.5:
							
							final_ind = skip ###This is the final slice index to be used for the consecutive block
							consecutive_segs.append([initial_ind,final_ind])
						initial_ind = skip + 1
					if len(skips) > .5 and initial_ind < len(dt_intervals)-1:
						consecutive_segs.append([initial_ind,len(dt_intervals)])
					if len(skips) < .5:
						consecutive_segs.append([0,len(dt_intervals)])
						
					rs_cons = []
					dts_cons = []
					
					for seg in consecutive_segs:
						rs_cons.extend( rs[seg[0]:seg[1]] )
						dts_cons.extend( dts[seg[0]:seg[1]] )
						
						coupling_list = extract_onestep_couplings( rs[seg[0]:seg[1]], thetas[seg[0]:seg[1]] )
					
						if len(coupling_list) > .5:
							if traj_ind not in couplings_by_cell[experiment][treatment][sample]:
								couplings_by_cell[experiment][treatment][sample][traj_ind] = []
							couplings_by_cell[experiment][treatment][sample][traj_ind].extend(coupling_list)
							
					speeds_by_cell[experiment][treatment][sample][traj_ind] = numpy.array(rs_cons)/numpy.array(dts_cons)*60 ###Speeds in microns/min, measured on consecutive steps (secant approx)
				
	return couplings_by_cell				
					
	
def subsample_timepoints( sample_trajectories, subsampling_factor ):
	
	first_traj = True
	new_sample_trajectories = {}
	for traj_ind in sample_trajectories:
		new_sample_trajectories[traj_ind] = []
		if first_traj:
			traj = numpy.array(sample_trajectories[traj_ind])
			time_interval = numpy.min(traj[1:,0] - traj[:-1,0])
			first_traj = False
			
		traj_list = sample_trajectories[traj_ind]
		for entry in traj_list:
			if int(entry[0]/time_interval) % subsampling_factor < .5:
				
				new_sample_trajectories[traj_ind].append( entry )
	
	return new_sample_trajectories

def subsample_timepoints_time( sample_trajectories, target_interval ):
	###This function retains timepoints that are closest to the target interval; for use with high frequency data
	first_traj = True
	new_sample_trajectories = {}
	for traj_ind in sample_trajectories:
		new_sample_trajectories[traj_ind] = []
		if first_traj:
			traj = numpy.array(sample_trajectories[traj_ind])
			time_interval = numpy.min(traj[1:,0] - traj[:-1,0])
			first_traj = False
			
		traj_list = sample_trajectories[traj_ind]
		for entry in traj_list:
			if entry[0] % target_interval < time_interval/2 + .01: ###Take the timepoint closest to (but larger than) the target interval
				
				new_sample_trajectories[traj_ind].append( entry )
	
	return new_sample_trajectories
	
def calculate_MSDs( sample_trajectories, tau, overlap = False ):
	
	msds = {}
	for traj_ind in sample_trajectories:
		
		track = numpy.array(sample_trajectories[traj_ind])
		
		i = 0
		j = 0
		if not overlap:
			while i < track.shape[0] and j < track.shape[0]:
				
				if numpy.abs(track[j,0] - track[i,0] - tau) < 6.5: ###Time interval is within 5 seconds of tau (which should be at least 90 sec)
					if traj_ind not in msds:
						msds[traj_ind] = []
					msds[traj_ind].append( (track[j,1] - track[i,1])**2 + (track[j,2] - track[i,2])**2 )
					i = j
					
				j += 1
		else:
			while i < track.shape[0] and j < track.shape[0]:
				
				if numpy.abs(track[j,0] - track[i,0] - tau) < 6.5: ###Time interval is within 5 seconds of tau (which should be at least 90 sec)
					if traj_ind not in msds:
						msds[traj_ind] = []
					msds[traj_ind].append( (track[j,1] - track[i,1])**2   + (track[j,2] - track[i,2])**2 )
					i += 1
					j = i
					
				j += 1
	return msds

###Not currently in use for figures (below)

def calculate_disps_x( sample_trajectories, tau, overlap = False ):
	
	msds = {}
	for traj_ind in sample_trajectories:
		
		track = numpy.array(sample_trajectories[traj_ind])
		
		i = 0
		j = 0
		if not overlap:
			while i < track.shape[0] and j < track.shape[0]:
				
				if numpy.abs(track[j,0] - track[i,0] - tau) < 6.5: ###Time interval is within 6.5 seconds of tau (which should be at least 90 sec)
					if traj_ind not in msds:
						msds[traj_ind] = []
					msds[traj_ind].append( (track[j,1] - track[i,1])**2 )#+ (track[j,2] - track[i,2])**2 )
					i = j
					
				j += 1
		else:
			while i < track.shape[0] and j < track.shape[0]:
				
				if numpy.abs(track[j,0] - track[i,0] - tau) < 6.5: ###Time interval is within 6.5 seconds of tau (which should be at least 90 sec)
					if traj_ind not in msds:
						msds[traj_ind] = []
					msds[traj_ind].append( (track[j,1] - track[i,1]) )#  + (track[j,2] - track[i,2])**2 )
					i += 1
					j = i
					
				j += 1
	return msds
	

	
def velocity_autocorrelation_2D( sample_trajectories_polar, tau, overlap = True ):
	
	autocorrs = {}
	for traj_ind in sample_trajectories_polar:
		track_data = sample_trajectories_polar[traj_ind]
		i = 0
		j = 0
		if not overlap:
			while i < len(track_data[0]) and j < len(track_data[0]):
				
				if numpy.abs(track_data[0][j] - track_data[0][i] - tau) < 6.5: ###Time interval is within 5 seconds of tau (which should be at least 90 sec)
					if traj_ind not in autocorrs:
						autocorrs[traj_ind] = []
					
					autocorr = track_data[1][j]*numpy.cos(track_data[2][j])*track_data[1][i]*numpy.cos(track_data[2][i]) + track_data[1][j]*numpy.sin(track_data[2][j])*track_data[1][i]*numpy.sin(track_data[2][i])
					
					autocorrs[traj_ind].append( autocorr )
					i = j
					
				j += 1
		else:
			while i < len(track_data[0]) and j < len(track_data[0]):
				
				if numpy.abs(track_data[0][j] - track_data[0][i] - tau) < 6.5: ###Time interval is within 5 seconds of tau (which should be at least 90 sec)
					if traj_ind not in autocorrs:
						autocorrs[traj_ind] = []
					
					autocorr = track_data[1][j]*numpy.cos(track_data[2][j])*track_data[1][i]*numpy.cos(track_data[2][i]) + track_data[1][j]*numpy.sin(track_data[2][j])*track_data[1][i]*numpy.sin(track_data[2][i])
					
					autocorrs[traj_ind].append( autocorr )
					i += 1
					
				j += 1
	return autocorrs
	
def velocity_autocorrelation_x( sample_trajectories, tau, overlap = True ):	
	
	autocorrs = {}
	for traj_ind in sample_trajectories:
		track_data = sample_trajectories[traj_ind]
		i = 0
		j = 0
		if not overlap:
			while i < len(track_data[0]) and j < len(track_data[0]):
				
				if numpy.abs(track_data[0][j] - track_data[0][i] - tau) < 6.5: ###Time interval is within 5 seconds of tau (which should be at least 90 sec)
					if traj_ind not in autocorrs:
						autocorrs[traj_ind] = []
					
					autocorr = track_data[1][j]*numpy.cos(track_data[2][j])*track_data[1][i]*numpy.cos(track_data[2][i])

					autocorrs[traj_ind].append( autocorr )
					i = j
					
				j += 1
		else:
			while i < len(track_data[0]) and j < len(track_data[0]):
				
				if numpy.abs(track_data[0][j] - track_data[0][i] - tau) < 6.5: ###Time interval is within 5 seconds of tau (which should be at least 90 sec)
					if traj_ind not in autocorrs:
						autocorrs[traj_ind] = []
					
					autocorr = track_data[1][j]*numpy.cos(track_data[2][j])*track_data[1][i]*numpy.cos(track_data[2][i])
					
					autocorrs[traj_ind].append( autocorr )
					i += 1
					
				j += 1
	return autocorrs

def speed_autocorrelation( sample_trajectories, tau, overlap = True):
	
	autocorrs = {}
	for traj_ind in sample_trajectories:
		track_data = sample_trajectories[traj_ind]
		i = 0
		j = 0
		if not overlap:
			while i < len(track_data[0]) and j < len(track_data[0]):
				
				if numpy.abs(track_data[0][j] - track_data[0][i] - tau) < 6.5: ###Time interval is within 5 seconds of tau (which should be at least 90 sec)
					if traj_ind not in autocorrs:
						autocorrs[traj_ind] = []
					
					autocorr = track_data[1][j]*track_data[1][i]
					autocorrs[traj_ind].append( autocorr )
					i = j
					
				j += 1
		else:
			while i < len(track_data[0]) and j < len(track_data[0]):
				
				if numpy.abs(track_data[0][j] - track_data[0][i] - tau) < 6.5: ###Time interval is within 5 seconds of tau (which should be at least 90 sec)
					if traj_ind not in autocorrs:
						autocorrs[traj_ind] = []

					
					autocorr = track_data[1][j]*track_data[1][i]
					
					autocorrs[traj_ind].append( autocorr )
					i += 1
					
				j += 1
	return autocorrs