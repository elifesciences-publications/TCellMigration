import trajectory_analysis_functions_09192019
import numpy
import matplotlib.pylab as pt
import seaborn as sns
from scipy.stats import binned_statistic
import matplotlib.gridspec as gridspec
from scipy import interpolate
from scipy.optimize import curve_fit
import matplotlib
from cycler import cycler
import simulate_spc_model
import scipy.special
from scipy.stats import spearmanr

def Lorentzian_with_noise(xvec, beta, alpha, sigma):
	
	return numpy.log10(alpha/(1 + (xvec*beta)**2) + 4*sigma**2*(1 - numpy.cos(numpy.pi*xvec/numpy.max(xvec))))

def Lorentzian_normed_with_noise(xvec, beta, sigma):
	
	return beta/(1 + (xvec*beta)**2) + 4*sigma**2*(1 - numpy.cos(numpy.pi*xvec/numpy.max(xvec)))
	
def Lorentzian(xvec, beta, alpha):
	
	return alpha/(1 + (xvec/beta)**2)# + 4*sigma**2*(1 - numpy.cos(numpy.pi*xvec/numpy.max(xvec)))

def piecewise_linear(x, x0, y0, k1, k2):
    return numpy.piecewise(x, [x < x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])

def linear_part2(x, x0, y0, k1, k2):
	return k2*x + y0-k2*x0

def linear_part1(x, x0, y0, k1, k2):
	return k1*x + y0-k1*x0
 
def log_gaussian(x, A, sigma, x0):
	return A - .5*(x - x0)**2/sigma**2
	
def log_gaussian2(x, A, sigma):
	return A - .5*(x)**2/sigma**2

def log_exp(x,A,b,g):
	return A - b*x**g

def log_gamma(x,a,lam):
	gamma_pdf = lam**a * x**(a-1) * numpy.exp(-lam*x) / scipy.special.gamma(a)
	return numpy.log(gamma_pdf)
	
def gaussian(x, A, sigma, x0):
	return A*numpy.exp(-.5*(x - x0)**2/sigma**2)

def WLC_MSD(taus, beta, alpha, sigma):
	
	return alpha*taus*beta*(1 + beta/taus*(numpy.exp(-1*taus/beta) - 1)) + sigma

def speed_dist_angle_dist_KS_test( speeds_dict, speed_angle_coupling_dict ):
	####Collect overall distributions of speeds and angles
	from scipy.stats import kstest
	import scipy.interpolate
	
	all_speeds = []
	all_angles = []
	
	speeds_by_cell = []
	angles_by_cell = []
	
	for experiment in speeds_dict:
		if 'fish' in experiment:
			for treatment in speeds_dict[experiment]:
				if 'control' in treatment and 'highfreq' not in treatment:
					for sample in speeds_dict[experiment][treatment]:
						for traj_ind in speeds_dict[experiment][treatment][sample]:
							
							if traj_ind in speed_angle_coupling_dict[experiment][treatment][sample] and len(speeds_dict[experiment][treatment][sample][traj_ind])>=30:
								
								
								relative_angle_data = speed_angle_coupling_dict[experiment][treatment][sample][traj_ind]
								relative_angles = numpy.array([entry[1] for entry in relative_angle_data])
								
								all_speeds.extend( speeds_dict[experiment][treatment][sample][traj_ind] )
								all_angles.extend( relative_angles )
								
								speeds_by_cell.append(speeds_dict[experiment][treatment][sample][traj_ind])
								angles_by_cell.append(relative_angles)
								#if numpy.max(speeds_dict[experiment][treatment][sample][traj_ind]) > 40:
									#print(sample, traj_ind)
	###Construct cdfs from speed and angle distributions
	all_speeds = numpy.array(all_speeds)
	all_angles = numpy.array(all_angles)
	
	print('Overall Speed distribution ntraj, ',len(all_speeds))
	print('Overall Angle distribution ntraj, ',len(all_angles))
	
	#print(numpy.max(all_speeds))
	speed_pdf,speed_binedges = numpy.histogram(all_speeds,bins=numpy.percentile(all_speeds,numpy.arange(0,101,4)),density=True)
	
	angles_pdf,angle_binedges = numpy.histogram(all_angles,bins=numpy.percentile(all_angles,numpy.arange(0,101,4)),density=True)
	
	speeds_cum = numpy.cumsum(speed_pdf)
	angles_cum = numpy.cumsum(angles_pdf)
	
	speeds_cum = numpy.insert(speeds_cum,0,0)/speeds_cum[-1]
	angles_cum = numpy.insert(angles_cum,0,0)/angles_cum[-1]
	
	speeds_cdf = scipy.interpolate.interp1d(speed_binedges,speeds_cum)
	angles_cdf = scipy.interpolate.interp1d(angle_binedges,angles_cum)
	pt.figure()
	pt.plot(speed_binedges,speeds_cum)
	pt.figure()
	pt.plot(angle_binedges,angles_cum)
	#pt.show()
	speed_pvals = []
	angle_pvals = []
	for i in range(len(speeds_by_cell)):
		D,p_s = scipy.stats.kstest(speeds_by_cell[i],speeds_cdf)
		D,p_a = scipy.stats.kstest(angles_by_cell[i],angles_cdf)
		
		speed_pvals.append(p_s)
		angle_pvals.append(p_a)
	#print(speed_pvals)
	speed_pvals = numpy.array(speed_pvals)
	angle_pvals = numpy.array(angle_pvals)
	
	print(numpy.sum(speed_pvals < .01)/len(speed_pvals))
	print(numpy.sum(angle_pvals < .01)/len(angle_pvals))
	
	
	
	
	
	
	
def define_speed_classes( speeds_dict, step_angle_coupling_dict, drift_corrected_traj_dict):


	###This is a helper function for all other functions that bin cells into the 5 speed quintiles (turn angle distribution; MSDs by speed class; local speed vs. turn angle)
	###It defines the set of cell indexes with at least 30 good timepoints and containing a measurement for all of the time intervals where MSD is measured, for each speed class.
	
	###Collect mean speeds to compute quintile speed classes
	
	n_tsteps_min = 30
	taus = [45,90,135,180,270,360,450,540,630,720,810,900,990,1080]
	mean_speed_list_all = []
	index_set = set()
	for experiment in step_angle_coupling_dict:
		if 'fish' in experiment:
			for treatment in step_angle_coupling_dict[experiment]:
				if 'control' in treatment and 'highfreq' not in treatment:
					
					for sample in step_angle_coupling_dict[experiment][treatment]:
						
						for traj_ind in step_angle_coupling_dict[experiment][treatment][sample]:
							
							if traj_ind in speeds_dict[experiment][treatment][sample]:
								#mean_speed=numpy.mean(speeds_dict[experiment][treatment][sample][traj_ind])
								mean_speed=numpy.mean(speeds_dict[experiment][treatment][sample][traj_ind])
								if numpy.isnan(mean_speed):
									print('wtf????',sample,traj_ind)
								if len(speeds_dict[experiment][treatment][sample][traj_ind]) >= n_tsteps_min:
									mean_speed_list_all.append(mean_speed)
									full_ind = sample + '-' + traj_ind
									index_set.add(full_ind)
									
	speed_classes = numpy.percentile(mean_speed_list_all,numpy.arange(0,101,20))
	msds_by_tau = {}
	for experiment in speeds_dict:
		if 'fish' in experiment:
			for treatment in speeds_dict[experiment]:
				if 'control' in treatment and 'highfreq' not in treatment:
					for sample in speeds_dict[experiment][treatment]:
						for tau in taus:
						
							msds_by_traj = trajectory_analysis_functions_09192019.calculate_MSDs(drift_corrected_traj_dict[experiment][treatment][sample], tau, overlap=True)
							
							for traj_ind in msds_by_traj:
								full_ind = sample + '-' + traj_ind
								if full_ind in index_set:
									mean_speed=numpy.mean(speeds_dict[experiment][treatment][sample][traj_ind])
									speed_class = numpy.argmax( mean_speed - speed_classes < 0 ) - 1
									
									if speed_class < 0:
										speed_class = len(speed_classes) - 1
									
									
									if speed_class not in msds_by_tau:
										msds_by_tau[speed_class] = {}
									if tau not in msds_by_tau[speed_class]:
										msds_by_tau[speed_class][tau] = {}
										
									msds_by_tau[speed_class][tau][full_ind] = msds_by_traj[traj_ind]
									
									
	
	selectable_inds_by_speed_class = {}
	
	for speed_class in range(5):
		
		inds0 = set(msds_by_tau[speed_class][taus[0]].keys())
		
		for tau in taus[1:]:
			inds0.intersection_update(set(msds_by_tau[speed_class][tau].keys()))
		selectable_inds_by_speed_class[speed_class] = numpy.array(list(inds0))
		print(speed_class,selectable_inds_by_speed_class[speed_class].shape)
	return selectable_inds_by_speed_class

def define_speed_classes_subsamp( speeds_dict, step_angle_coupling_dict, drift_corrected_traj_dict):

	###This is a helper function for all other functions that bin cells into the 5 speed quintiles (turn angle distribution; MSDs by speed class; local speed vs. turn angle)
	###It defines the set of cell indexes with at least 30 good timepoints and containing a measurement for all of the time intervals where MSD is measured, for each speed class.
	
	###Collect mean speeds to compute quintile speed classes
	
	n_tsteps_min = 15
	taus = [90,180,270,360,450,540,630,720,810,900,990,1080]
	mean_speed_list_all = []
	index_set = set()
	for experiment in step_angle_coupling_dict:
		if 'fish' in experiment:
			for treatment in step_angle_coupling_dict[experiment]:
				if 'control' in treatment and 'highfreq' not in treatment:
					
					for sample in step_angle_coupling_dict[experiment][treatment]:
						
						for traj_ind in step_angle_coupling_dict[experiment][treatment][sample]:
							
							if traj_ind in speeds_dict[experiment][treatment][sample]:
								#mean_speed=numpy.mean(speeds_dict[experiment][treatment][sample][traj_ind])
								mean_speed=numpy.mean(speeds_dict[experiment][treatment][sample][traj_ind])
								if numpy.isnan(mean_speed):
									print('wtf????',sample,traj_ind)
								if len(speeds_dict[experiment][treatment][sample][traj_ind]) >= n_tsteps_min:
									mean_speed_list_all.append(mean_speed)
									full_ind = sample + '-' + traj_ind
									index_set.add(full_ind)
									
	speed_classes = numpy.percentile(mean_speed_list_all,numpy.arange(0,101,20))
	msds_by_tau = {}
	for experiment in speeds_dict:
		if 'fish' in experiment:
			for treatment in speeds_dict[experiment]:
				if 'control' in treatment and 'highfreq' not in treatment:
					for sample in speeds_dict[experiment][treatment]:
						for tau in taus:
						
							msds_by_traj = trajectory_analysis_functions_09192019.calculate_MSDs(drift_corrected_traj_dict[experiment][treatment][sample], tau, overlap=True)
							
							for traj_ind in msds_by_traj:
								full_ind = sample + '-' + traj_ind
								if full_ind in index_set:
									mean_speed=numpy.mean(speeds_dict[experiment][treatment][sample][traj_ind])
									speed_class = numpy.argmax( mean_speed - speed_classes < 0 ) - 1
									
									if speed_class < 0:
										speed_class = len(speed_classes) - 1
									
									
									if speed_class not in msds_by_tau:
										msds_by_tau[speed_class] = {}
									if tau not in msds_by_tau[speed_class]:
										msds_by_tau[speed_class][tau] = {}
										
									msds_by_tau[speed_class][tau][full_ind] = msds_by_traj[traj_ind]
									
									
	
	selectable_inds_by_speed_class = {}
	
	for speed_class in range(5):
		
		inds0 = set(msds_by_tau[speed_class][taus[0]].keys())
		
		for tau in taus[1:]:
			inds0.intersection_update(set(msds_by_tau[speed_class][tau].keys()))
		selectable_inds_by_speed_class[speed_class] = numpy.array(list(inds0))
		print(speed_class,selectable_inds_by_speed_class[speed_class].shape)
	return selectable_inds_by_speed_class
	
def turn_angles_by_speed_class( step_angle_coupling_dict, speeds_dict, selectable_inds_by_speed_class, ax ):
	
	turn_angles = {}
	mean_speeds = {}
	
	n_bootstrap = 200
	
	for speed_class in range(5):
		if speed_class not in turn_angles:
			
			turn_angles[speed_class] = {}
			mean_speeds[speed_class] = []
		for experiment in step_angle_coupling_dict:
			for treatment in step_angle_coupling_dict[experiment]:
				if 'control' in treatment:
					for full_ind in selectable_inds_by_speed_class[speed_class]:
					
						sample,traj_ind = full_ind.split('-')
					
						if sample in speeds_dict[experiment][treatment]:
							#print(experiment,treatment,sample,traj_ind)
							mean_speed=numpy.mean(speeds_dict[experiment][treatment][sample][traj_ind])
							turn_angles[speed_class][full_ind] = [entry[1] for entry in step_angle_coupling_dict[experiment][treatment][sample][traj_ind]]
							mean_speeds[speed_class].append(mean_speed)
									
	
	angle_bins = numpy.arange(0,numpy.pi + .01,numpy.pi/6.)
	
	angle_dists_bootstrapped = {}
	
	speed_label_list = []
	
	for speed_class in range(5):
		
		n_cells = len(selectable_inds_by_speed_class[speed_class])
		index_list_selectable = numpy.array(selectable_inds_by_speed_class[speed_class])
		angle_dists_bootstrapped[speed_class] = []
		speed_label_list.append( str(numpy.round(numpy.mean(mean_speeds[speed_class]),1)) + r' $\frac{\mu m}{min}$')
		for n in range(n_bootstrap):
		
			cell_choices = numpy.random.randint(0,high=n_cells,size=n_cells)
		
			indexes = index_list_selectable[cell_choices]
			turn_angles_list = []
			for index in indexes:
				
				turn_angles_list.extend( turn_angles[speed_class][index] )
			
			hist,bins = numpy.histogram(turn_angles_list, bins=angle_bins, density=True)
			
			angle_dists_bootstrapped[speed_class].append(hist)
	
	offset = .8*numpy.pi/6/6
	
	for speed_class in range(5):
		
		bs_data = numpy.array(angle_dists_bootstrapped[speed_class])
		
		medians = numpy.percentile(bs_data,50,axis=0)
		lb = numpy.percentile(bs_data,5,axis=0)
		ub = numpy.percentile(bs_data,95,axis=0)
		
		ax.bar(angle_bins[:-1] + speed_class*offset, medians, yerr = [medians - lb, ub - medians], width = .9*offset)
	
	lgnd = ax.legend(speed_label_list,fontsize=8,ncol=2,handletextpad=.2)
	
	for handle in lgnd.legendHandles:
		handle.set_width(6)
		handle.set_height(6)
	ax.set_xlabel('Turn angle (rad)')
	ax.set_ylabel('Probability density')



def speed_angle_coupling_by_speed_class( step_angle_coupling_dict, speeds_dict, interval, selectable_inds_by_speed_class, ax ):
	
	turn_angles = {}
	local_speeds = {}
	mean_speeds = {}
	
	n_bootstrap = 200
	
	for speed_class in range(5):
		if speed_class not in turn_angles:
			
			turn_angles[speed_class] = {}
			mean_speeds[speed_class] = []
			local_speeds[speed_class] = {}
		for experiment in step_angle_coupling_dict:
			for treatment in step_angle_coupling_dict[experiment]:
				if 'control' in treatment:
					for full_ind in selectable_inds_by_speed_class[speed_class]:
					
						sample,traj_ind = full_ind.split('-')
					
						if sample in speeds_dict[experiment][treatment]:
							#print(experiment,treatment,sample,traj_ind)
							mean_speed=numpy.mean(speeds_dict[experiment][treatment][sample][traj_ind])
							turn_angles[speed_class][full_ind] = [entry[1] for entry in step_angle_coupling_dict[experiment][treatment][sample][traj_ind]]
							mean_speeds[speed_class].append(mean_speed)
							local_speeds[speed_class][full_ind] = [numpy.mean(entry[0])/interval*60 for entry in step_angle_coupling_dict[experiment][treatment][sample][traj_ind]]

	angle_bins = numpy.arange(0,numpy.pi + .01,numpy.pi/6.)
	
	angle_stat_bootstrapped = {}
	xlocs_bootstrapped = {}
	
	speed_label_list = []
	
	speed_bins = numpy.arange(0,15,2)
	
	for speed_class in range(5):
		
		index_list_selectable = selectable_inds_by_speed_class[speed_class]
		n_cells = len(index_list_selectable)
		print(n_cells)
		angle_stat_bootstrapped[speed_class] = []
		xlocs_bootstrapped[speed_class] = []
		speed_label_list.append( str(numpy.round(numpy.mean(mean_speeds[speed_class]),1)) + r' $\frac{\mu m}{min}$')
		for n in range(n_bootstrap):
			
			cell_choices = numpy.random.randint(0,high=n_cells,size=n_cells)
		
			indexes = index_list_selectable[cell_choices]
			
			bootstrapped_angles = []
			bootstrapped_local_speeds = []
			
			for index in indexes:
					
				bootstrapped_angles.extend(turn_angles[speed_class][index])
				bootstrapped_local_speeds.extend(local_speeds[speed_class][index])
			
			binned_measure,bins,nbins = binned_statistic(bootstrapped_local_speeds,numpy.cos(bootstrapped_angles),bins=speed_bins)
			xlocs_measure,bins,nbins = binned_statistic(bootstrapped_local_speeds,bootstrapped_local_speeds,bins=speed_bins)
			
			angle_stat_bootstrapped[speed_class].append(binned_measure)
			xlocs_bootstrapped[speed_class].append(xlocs_measure)
	
	for speed_class in range(5):
		
		medians = numpy.percentile(angle_stat_bootstrapped[speed_class],50,axis=0)
		lb = numpy.percentile(angle_stat_bootstrapped[speed_class],5,axis=0)
		ub = numpy.percentile(angle_stat_bootstrapped[speed_class],95,axis=0)
		median_xlocs = numpy.percentile( xlocs_bootstrapped[speed_class],50,axis=0)
		
		ax.errorbar(median_xlocs,medians,yerr=[medians-lb,ub-medians],marker='o',elinewidth=.5,markersize=4)
	
	ax.set_xlabel('Local speed ($\mu$m/min)')
	ax.set_ylabel(r'$\langle cos\theta \rangle$')
	ax.set_ylim(-.4,.8)
	lgnd = ax.legend(speed_label_list,fontsize=8,ncol=2,loc='upper left',frameon=False)#loc='lower left',bbox_to_anchor=(.26,.01))#,loc="upper left",frameon=False)

def boutx_distribution( speeds_dict,trajectory_dict_polar_interp,ax):
	n_tsteps_min = 30
	n_bootstrap = 200
	bout_length_dict = trajectory_analysis_functions_09192019.bout_lengthsx( trajectory_dict_polar_interp )
	bout_lens_scaled_all = []
	index_list = []
	bout_dict = {}
	for experiment in bout_length_dict:
		if 'fish' in experiment:
			for treatment in bout_length_dict[experiment]:
				if 'control' in treatment:
					
					for sample in bout_length_dict[experiment][treatment]:
						
						for traj_ind in bout_length_dict[experiment][treatment][sample]:
							if len(speeds_dict[experiment][treatment][sample]) >= n_tsteps_min:
								bout_lens_scaled = numpy.array(bout_length_dict[experiment][treatment][sample][traj_ind])/numpy.mean(bout_length_dict[experiment][treatment][sample][traj_ind])
								bout_lens_scaled_all.extend(bout_lens_scaled)
								index_list.append(sample + '_' + traj_ind)
								bout_dict[sample + '_' + traj_ind] = bout_lens_scaled
	histbins = numpy.percentile(bout_lens_scaled_all,numpy.arange(0,101,5))
	histvals,binedges = numpy.histogram(bout_lens_scaled_all,bins=histbins,density=True)
	xlocs,binedges,nbins = binned_statistic(bout_lens_scaled_all,bout_lens_scaled_all,bins=histbins)
	selectable_indexes = numpy.array(index_list)
	hist_stat_bs = []
	for n in range(n_bootstrap):
		indexes = numpy.random.choice(selectable_indexes,size=len(selectable_indexes))
		bl_bs = []
		for ind in indexes:
			bl_bs.extend(bout_dict[ind])
		histvals_bs,binedges = numpy.histogram(bl_bs,bins=histbins,density=True)
		hist_stat_bs.append(histvals_bs)
	cf,res = curve_fit(log_exp, xlocs[:-1], numpy.log(histvals[:-1]))
	print(cf)
	hist_stat_bs = numpy.array(hist_stat_bs)
	lb = numpy.percentile(hist_stat_bs,5,axis=0)
	ub = numpy.percentile(hist_stat_bs,95,axis=0)
	ax.set_xscale('log')
	ax.set_yscale('log')
	ax.errorbar(xlocs,histvals,yerr=[histvals-lb,ub-histvals],marker='o',linestyle='None',markersize=4)
	handle,=ax.plot(xlocs,numpy.exp(log_exp(xlocs,*cf)),'k')
	ax.set_xlabel(r'Displacement deviation')
	ax.set_ylabel(r'Probability density')
	ax.legend([handle],[r'Aexp($-\gamma x^\beta)$'])
	print('beta, ',str(round(cf[2],1)))
	print('bout dist nbouts, ',len(bout_lens_scaled_all))

def all_cells_manifold(step_angle_coupling_dict, speeds_dict, ax ):

	step_size_angle_coupling_by_speed = {}

	mean_speeds = {}
	mean_cosangles = {}
	mean_couplings = {}
	
	cosangles_std_err = {}
	speeds_std_err = {}

	for experiment in step_angle_coupling_dict:
		mean_couplings[experiment] = {}
		mean_speeds[experiment] = {}
		mean_cosangles[experiment] = {}
		
		cosangles_std_err[experiment] = {}
		speeds_std_err[experiment] = {}
		for treatment in step_angle_coupling_dict[experiment]:
			mean_couplings[experiment][treatment] = []
			mean_speeds[experiment][treatment] = []
			mean_cosangles[experiment][treatment] = []
			
			cosangles_std_err[experiment][treatment] = []
			speeds_std_err[experiment][treatment] = []
			for sample in step_angle_coupling_dict[experiment][treatment]:
				for traj_ind in step_angle_coupling_dict[experiment][treatment][sample]:
					if traj_ind in speeds_dict[experiment][treatment][sample]:
						
						mean_speed=numpy.mean(speeds_dict[experiment][treatment][sample][traj_ind])
						speed_std_err = numpy.std(speeds_dict[experiment][treatment][sample][traj_ind])/numpy.sqrt(len(speeds_dict[experiment][treatment][sample][traj_ind]) - 1)
						
						if ~numpy.isnan(mean_speed) and len(step_angle_coupling_dict[experiment][treatment][sample][traj_ind]) > 30:
							coupling_data = step_angle_coupling_dict[experiment][treatment][sample][traj_ind]
							step_sizes = [numpy.mean(entry[0]) for entry in coupling_data]
							angles = [numpy.cos(entry[1]) for entry in coupling_data]
							
							angles_std_err = numpy.std(angles)/numpy.sqrt(len(angles) - 1)
							
							corr = numpy.corrcoef(step_sizes,angles)[0,1]
								
							mean_speeds[experiment][treatment].append(mean_speed)
							mean_cosangles[experiment][treatment].append(numpy.mean(angles))
							mean_couplings[experiment][treatment].append(corr)
								
							
							cosangles_std_err[experiment][treatment].append(angles_std_err)
							speeds_std_err[experiment][treatment].append(speed_std_err)

	for experiment in mean_speeds:
		if 'fish' in experiment:
			for treatment in mean_speeds[experiment]:
				if 'control' in treatment and 'highfreq' not in treatment:
	
					upper = numpy.percentile(mean_couplings[experiment][treatment],98)
					lower = numpy.percentile(mean_couplings[experiment][treatment],3)
					sct = ax.scatter(mean_speeds[experiment][treatment],mean_cosangles[experiment][treatment],s=2,alpha=.6,c=mean_couplings[experiment][treatment],vmin=lower,vmax=upper,cmap='viridis')
					cbar = pt.colorbar(sct,ax=ax)
					#my_cmap = matplotlib.cm.get_cmap(cmap='RdBu',vmin=-1*upper,vmax=upper)
					my_norm = matplotlib.colors.Normalize(vmin=lower, vmax=upper, clip=False)
					my_mappable = matplotlib.cm.ScalarMappable(cmap='viridis',norm=my_norm)
					_, __ ,errorlinecollection = ax.errorbar(mean_speeds[experiment][treatment],mean_cosangles[experiment][treatment],xerr=speeds_std_err[experiment][treatment],yerr=cosangles_std_err[experiment][treatment],marker='',linestyle='None',zorder=0,alpha=.4,linewidth=.1)
					error_color = my_mappable.to_rgba(mean_couplings[experiment][treatment])

					errorlinecollection[0].set_color(error_color)
					errorlinecollection[1].set_color(error_color)
			
	cbar.set_label(r'Corr($cos\mathrm{\theta }$,s)')
	cbar.ax.tick_params(labelsize=8)
	ax.set_xlabel(r'$\langle s \rangle_{\mathrm{cell}}$')
	ax.set_ylabel(r'$\langle cos\mathrm{\theta} \rangle_{\mathrm{cell}}$')

def all_cells_manifold_3d(step_angle_coupling_dict, speeds_dict, ax ):

	step_size_angle_coupling_by_speed = {}

	mean_speeds = {}
	mean_cosangles = {}
	mean_couplings = {}
	
	cosangles_std_err = {}
	speeds_std_err = {}
	
	focal_sample = 'lckgfp_dob07072019_fish3_488_trial3_3' 

	chosen_indexes = ['24','30','11','15']
	
	color_cycle = {'24':'goldenrod','15':'C6','30':'C9','11':'C5'}
	highlighted_traj_pos = {}
	for experiment in step_angle_coupling_dict:
		mean_couplings[experiment] = {}
		mean_speeds[experiment] = {}
		mean_cosangles[experiment] = {}
		
		cosangles_std_err[experiment] = {}
		speeds_std_err[experiment] = {}
		for treatment in step_angle_coupling_dict[experiment]:
			mean_couplings[experiment][treatment] = []
			mean_speeds[experiment][treatment] = []
			mean_cosangles[experiment][treatment] = []
			
			cosangles_std_err[experiment][treatment] = []
			speeds_std_err[experiment][treatment] = []
			for sample in step_angle_coupling_dict[experiment][treatment]:
				for traj_ind in step_angle_coupling_dict[experiment][treatment][sample]:
					if traj_ind in speeds_dict[experiment][treatment][sample] and len(speeds_dict[experiment][treatment][sample][traj_ind]) > .5:
						
						mean_speed=numpy.mean(speeds_dict[experiment][treatment][sample][traj_ind])
						speed_std_err = numpy.std(speeds_dict[experiment][treatment][sample][traj_ind])/numpy.sqrt(len(speeds_dict[experiment][treatment][sample][traj_ind]) - 1)
						
						if ~numpy.isnan(mean_speed) and len(step_angle_coupling_dict[experiment][treatment][sample][traj_ind]) > 20:
							coupling_data = step_angle_coupling_dict[experiment][treatment][sample][traj_ind]
							step_sizes = [numpy.mean(entry[0]) for entry in coupling_data]
							angles = [numpy.cos(entry[1]) for entry in coupling_data]
							
							angles_std_err = numpy.std(angles)/numpy.sqrt(len(angles) - 1)
							
							corr = numpy.corrcoef(step_sizes,angles)[0,1]
								
							if sample == focal_sample and traj_ind in chosen_indexes:
								highlighted_traj_pos[traj_ind] = (mean_speed,numpy.mean(angles),corr)
								print(focal_sample,traj_ind)
								
							else:
								mean_speeds[experiment][treatment].append(mean_speed)
								mean_cosangles[experiment][treatment].append(numpy.mean(angles))
								mean_couplings[experiment][treatment].append(corr)
								cosangles_std_err[experiment][treatment].append(angles_std_err)
								speeds_std_err[experiment][treatment].append(speed_std_err)

	for experiment in mean_speeds:
		if 'fish' in experiment:
			for treatment in mean_speeds[experiment]:
				if 'control' in treatment and 'highfreq' not in treatment:
	
					upper = numpy.percentile(mean_couplings[experiment][treatment],98)
					lower = numpy.percentile(mean_couplings[experiment][treatment],3)
					x = mean_speeds[experiment][treatment]
					y = mean_cosangles[experiment][treatment]
					z = mean_couplings[experiment][treatment]
					sct = ax.scatter(x,y,z,s=3,alpha=.6,c='C0')
					
					#ax.plot(x, z, 'o', color='grey', zdir='y', zs=.8)
					#ax.plot(y, z, 'o', color='grey', zdir='x', zs=0)
					ax.plot(x, y, 'o', color='grey', zdir='z', zs=-.4, markersize=2, alpha=.6)
					
					#cbar = pt.colorbar(sct,ax=ax)
					#my_cmap = matplotlib.cm.get_cmap(cmap='RdBu',vmin=-1*upper,vmax=upper)
					#my_norm = matplotlib.colors.Normalize(vmin=lower, vmax=upper, clip=False)
					#my_mappable = matplotlib.cm.ScalarMappable(cmap='viridis',norm=my_norm)
					#_, __ ,errorlinecollection = ax.errorbar(mean_speeds[experiment][treatment],mean_cosangles[experiment][treatment],xerr=speeds_std_err[experiment][treatment],yerr=cosangles_std_err[experiment][treatment],marker='',linestyle='None',zorder=0,alpha=.4,linewidth=.1)
					#error_color = my_mappable.to_rgba(mean_couplings[experiment][treatment])

					#errorlinecollection[0].set_color(error_color)
					#errorlinecollection[1].set_color(error_color)
			
	#cbar.set_label(r'Corr($cos\mathrm{\theta }$,s)')
	#cbar.ax.tick_params(labelsize=8)
	
	for traj_ind in chosen_indexes:
		x,y,z = highlighted_traj_pos[traj_ind]
		ax.scatter(x,y,z,s=14,c=color_cycle[traj_ind],zorder=2)
		ax.scatter(x,y,-.4,'o',color=color_cycle[traj_ind],s=14, zorder=2)
		#ax.plot(x,y,'o',color=color_cycle[traj_ind],zdir='z', zs=-.4, markersize=2, alpha=.6)
		
	ax.set_xlabel(r'$\langle s \rangle_{\mathrm{cell}}$')
	ax.set_ylabel(r'$\langle cos\mathrm{\theta} \rangle_{\mathrm{cell}}$')
	ax.set_zlabel(r'Corr($cos\mathrm{\theta }$,s)')

def manifold_anova(step_angle_coupling_dict, speeds_dict, ax, ax2 ):
	####What fraction of the variance that is not accounted for by the noise is accounted for by the 'dependent' variable?
	from scipy.interpolate import UnivariateSpline
	
	step_size_angle_coupling_by_speed = {}

	mean_speeds = {}
	mean_cosangles = {}
	mean_couplings = {}
	
	cosangles_var = {}
	speeds_std_err = {}
	
	cosangles_all = []
	cosangles_all_var = []
	speeds_all = []
	couplings_all_vars = []
	couplings_all = []
	for experiment in step_angle_coupling_dict:
		mean_couplings[experiment] = {}
		mean_speeds[experiment] = {}
		mean_cosangles[experiment] = {}
		
		cosangles_var[experiment] = {}
		speeds_std_err[experiment] = {}
		for treatment in step_angle_coupling_dict[experiment]:
			mean_couplings[experiment][treatment] = []
			mean_speeds[experiment][treatment] = []
			mean_cosangles[experiment][treatment] = []
			
			cosangles_var[experiment][treatment] = []
			speeds_std_err[experiment][treatment] = []
			for sample in step_angle_coupling_dict[experiment][treatment]:
				for traj_ind in step_angle_coupling_dict[experiment][treatment][sample]:
					if traj_ind in speeds_dict[experiment][treatment][sample]:
						
						mean_speed=numpy.mean(speeds_dict[experiment][treatment][sample][traj_ind])
						speed_std_err = numpy.std(speeds_dict[experiment][treatment][sample][traj_ind])/numpy.sqrt(len(speeds_dict[experiment][treatment][sample][traj_ind]) - 1)
						
						if ~numpy.isnan(mean_speed) and len(step_angle_coupling_dict[experiment][treatment][sample][traj_ind]) > 20:
							coupling_data = step_angle_coupling_dict[experiment][treatment][sample][traj_ind]
							step_sizes = [numpy.mean(entry[0]) for entry in coupling_data]
							angles = [numpy.cos(entry[1]) for entry in coupling_data]
							
							angles_std_err = numpy.std(angles)/numpy.sqrt(len(angles) - 1)
							angles_var = numpy.var(angles)/(len(angles) - 1)
							corr = numpy.corrcoef(step_sizes,angles)[0,1]
								
							
							mean_speeds[experiment][treatment].append(mean_speed)
							mean_cosangles[experiment][treatment].append(numpy.mean(angles))
							mean_couplings[experiment][treatment].append(corr)
							cosangles_var[experiment][treatment].append(angles_var)
							speeds_std_err[experiment][treatment].append(speed_std_err)
							if 'fish' in experiment and 'control' in treatment and 'highfreq' not in treatment:
								speeds_all.append(mean_speed)
								cosangles_all.append(numpy.mean(angles))
								cosangles_all_var.append(angles_var)
								couplings_all.append(corr)
								
	####Predict cosangles from speeds
	speeds_all = numpy.array(speeds_all)
	cosangles_all = numpy.array(cosangles_all)
	cosangles_all_var = numpy.array(cosangles_all_var)
	couplings_all = numpy.array(couplings_all)
	print('Manifold anova ntraj, ',len(speeds_all))
	sort_inds = numpy.argsort(speeds_all)
	params = numpy.polyfit(speeds_all[sort_inds], cosangles_all[sort_inds], deg=1)
	spl = UnivariateSpline(speeds_all[sort_inds], cosangles_all[sort_inds])
	
	pred_y = spl(speeds_all[sort_inds])
	#pred_y = params[1] + params[0]*speeds_all
	print(numpy.var(pred_y)/numpy.var(cosangles_all), numpy.mean(cosangles_all_var)/numpy.var(cosangles_all))
	
	spl_corr = UnivariateSpline(speeds_all[sort_inds], couplings_all[sort_inds])
	pred_y_corr = spl_corr(speeds_all[sort_inds])
	print(numpy.var(pred_y_corr)/numpy.var(couplings_all))
	xvec = numpy.arange(numpy.min(speeds_all),numpy.max(speeds_all),.05)
	ax.errorbar(speeds_all,cosangles_all,marker='o',linestyle='None',markersize=3,alpha=.5)#yerr=numpy.sqrt(cosangles_all_var),
	ax.plot(xvec,spl(xvec),'k')
	#ax.plot(xvec,params[1]+params[0]*xvec,'k')
	ax.set_xlabel(r'Cell speed ($\mu$m/min)')
	ax.set_ylabel(r'$\langle cos\mathrm{\theta} \rangle_{\mathrm{cell}}$')
	
	ax2.pie([numpy.var(pred_y)/numpy.var(cosangles_all), numpy.mean(cosangles_all_var)/numpy.var(cosangles_all),1-numpy.var(pred_y)/numpy.var(cosangles_all)- numpy.mean(cosangles_all_var)/numpy.var(cosangles_all)],labels=['Speed','Stochasticity','Other'],autopct='%1.1f%%')
	
def persistence_time_binned_fishonly(corr_times_dict, speeds_dict, ax):
	
	def linmod(x,a,b):
		return a*x + b
	def flat_mod(x,a):
		return a*numpy.ones((len(x),))
	corr_times_coll = {}
	speeds = []
	index_list = []
	
	corr_times_bootstrapped = []
	speeds_bootstrapped = []
	
	nbootstrap = 500
	
	for experiment in corr_times_dict:
		if 'fish' in experiment:
			for treatment in corr_times_dict[experiment]:
				if 'control' in treatment and 'highfreq' not in treatment:
					
					for sample in corr_times_dict[experiment][treatment]:
						
						for traj_ind in corr_times_dict[experiment][treatment][sample]:
							
							if traj_ind in speeds_dict[experiment][treatment][sample] and len(speeds_dict[experiment][treatment][sample][traj_ind]) > .5:
								
								mean_speed=numpy.mean(speeds_dict[experiment][treatment][sample][traj_ind])
								#print(speeds_dict[experiment][treatment][sample][traj_ind])
								#if ~numpy.isnan(mean_speed):
								full_index = sample + '_' + traj_ind	
								corr_times_coll[full_index] = corr_times_dict[experiment][treatment][sample][traj_ind]
								index_list.append(full_index)
								speeds.append(mean_speed)
	
	index_list_selectable = numpy.array(index_list)
	speeds_selectable = numpy.array(speeds)
	
	n_cells = len(index_list)
	speed_bins = numpy.percentile(speeds, numpy.arange(0,101,10))
	xlocs,bins,nbins = binned_statistic(speeds,speeds,bins=speed_bins)
	
	corr_times_stat_bs = []
	
	for n in range(nbootstrap):
		cell_choices = numpy.random.randint(0,high=n_cells,size=n_cells)
		
		indexes_bs = index_list_selectable[cell_choices]
		speeds_bs = speeds_selectable[cell_choices]
		#speeds_bs = []
		corr_times_bs = []
		for i in range(len(indexes_bs)):
			index = indexes_bs[i]
			ms = speeds_selectable[cell_choices[i]]
			#corr_times_bs.extend(numpy.array(corr_times_coll[index])/60. )
			#speeds_bs.extend([ms for it in range(len(corr_times_coll[index]))])
			corr_times_bs.append( numpy.mean(corr_times_coll[index])/60. )
		
		corr_time_stat,bins,nbins = binned_statistic(speeds_bs,corr_times_bs,bins=speed_bins)
		
		corr_times_stat_bs.append(corr_time_stat)
	
	corr_times_stat_bs = numpy.array(corr_times_stat_bs)
	
	corr_times_med = numpy.percentile(corr_times_stat_bs,50,axis=0)
	lb = numpy.percentile(corr_times_stat_bs,5,axis=0)
	ub = numpy.percentile(corr_times_stat_bs,95,axis=0)
	print('Persistence time measurement ntraj, ',len(speeds))
	###
	P_fit, pcov = curve_fit(linmod,xlocs,corr_times_med,sigma=ub-lb)
	P_fit_prw, pcov_prw = curve_fit(flat_mod,xlocs,corr_times_med,p0=[1.6],sigma=ub-lb,bounds=(0,numpy.inf))
	
	ax.errorbar(xlocs, corr_times_med, yerr=[corr_times_med - lb,ub-corr_times_med],linestyle='None',marker='o',elinewidth=.5,markersize=5)
	handle1,=ax.plot(xlocs,linmod(xlocs,*P_fit),'k')
	handle2,=ax.plot(xlocs,numpy.mean(corr_times_med)*numpy.ones((len(xlocs),)),'--',color='grey',linewidth=.7)
	ax.legend([handle1,handle2],['SPC','UPT'],loc="upper left")
	ax.set_xlabel('Cell speed ($\mu$m/min)')
	ax.set_ylabel('Persistence time (min)')

def PSDs_overall(corr_times_dict, trajectory_dict_polar, ax):
	
	experiment_list = ['fish_T']
	experiment_titles = ['D. rerio','M. musculus','Dictyostelium']

	study_labels = ['This study','Gerard et al.','Dang et al.']
	treatment_label_dict = {'control':'Control','rockout':'12 $\mu$M Rockout','Myo1g_KO':'Myo1g KO','Arp_KO':'Arpin KO','Arp_rescue':'Arpin rescue','WT':'WT'}
	experiment_ind = 0

	tmax = 30
	traj_list = []
	mean_speed_list = []

	ttot = 400
	hz_frequencies = 2*numpy.pi*(1/45)*numpy.arange(int(ttot/2))*1/ttot

	for experiment in experiment_list:
		
		treatment_list = []
		treatment_label_list = []
		for treatment in corr_times_dict[experiment]:
			if 'control' in treatment and 'highfreq' not in treatment:
				for sample in trajectory_dict_polar[experiment][treatment]:
					
					for traj_ind in trajectory_dict_polar[experiment][treatment][sample]:
						
						traj_data = numpy.array(trajectory_dict_polar[experiment][treatment][sample][traj_ind]).T
						
						if (traj_data[-1,0] - traj_data[0,0])/45 > tmax and numpy.sum(numpy.isnan(traj_data)) < .5:
							traj_list.append((sample,traj_ind))
							mean_speed_list.append(numpy.mean(traj_data[:,1]))
			
				len_pvec = int(ttot/2)
				
				
				power_list = []
				power_norm_list = []
				powerx_list = []
				norm_factor_list = []
				var_list = []
				for traj_tuple in traj_list:
					
					sample = traj_tuple[0]
					traj_ind = traj_tuple[1]
					
					interval = .75
					
					#interval_freqs = hz_frequencies*interval
				
					traj_data = numpy.array(trajectory_dict_polar[experiment][treatment][sample][traj_ind]).T
					
					times = traj_data[:,0]/45
					vx = traj_data[:,1]*numpy.cos(traj_data[:,2])/interval
					vy = traj_data[:,1]*numpy.sin(traj_data[:,2])/interval
					
					vx_norm = numpy.cos(traj_data[:,2])
					vy_norm = numpy.sin(traj_data[:,2])
					
					T = int(times[-1]-times[0])
					
					times = (traj_data[:,0] - traj_data[0,0])/45 ####Start all trajectories at t=0; ft is invariant under a translation in time
							
							
					###in case there are missing timepoints, interpolate the speeds vector from tp0 to tptmax
						
					#temp_vx = interpolate.interp1d(times,vx)
					#temp_vy = interpolate.interp1d(times,vy)
							
					vx_vec = vx
					vy_vec = vy
							
					####Zero pad out to ttot timepoints for shorter trajectories. (Note that we will need to be careful with the normalization to preserve the correct amplitude)
							
					if T < ttot:
						nadd = ttot - T
						vx_vec = numpy.concatenate((vx_vec, numpy.zeros((nadd,)) ))
						vy_vec = numpy.concatenate((vy_vec, numpy.zeros((nadd,)) ))
								
					vx_fft = numpy.fft.fft(vx_vec)
					vy_fft = numpy.fft.fft(vy_vec)	
					
					powerx = numpy.zeros((len_pvec,),dtype='float')
							
					powerx[0] = 1/ttot**2*2*ttot/T*numpy.absolute(vx_fft[0])**2
					powerx[len_pvec-1] = 1/ttot**2*2*ttot/T*numpy.absolute(vx_fft[len_pvec-1])**2
							
					for k in range(1,len_pvec-1):
						powerx[k] = 1/ttot**2*ttot/T*(numpy.absolute(vx_fft[k])**2 + numpy.absolute(vx_fft[ttot-k])**2)
					
					powery = numpy.zeros((len_pvec,),dtype='float')
							
					powery[0] = 1/ttot**2*2*ttot/T*numpy.absolute(vy_fft[0])**2
					powery[len_pvec-1] = 1/ttot**2*2*ttot/T*numpy.absolute(vy_fft[len_pvec-1])**2
							
					for k in range(1,len_pvec-1):
						powery[k] = 1/ttot**2*ttot/T*(numpy.absolute(vy_fft[k])**2 + numpy.absolute(vy_fft[ttot-k])**2)
					
					power_list.append(powerx+powery)
					norm_factor_list.append(ttot/T)
					
					####
					if T < ttot:
						nadd = ttot - T
						vx_norm = numpy.concatenate((vx_norm, numpy.zeros((nadd,)) ))
						vy_norm = numpy.concatenate((vy_norm, numpy.zeros((nadd,)) ))
								
					vx_fft_norm = numpy.fft.fft(vx_norm)
					vy_fft_norm = numpy.fft.fft(vy_norm)	
						
					powerx_norm = numpy.zeros((len_pvec,),dtype='float')
							
					powerx_norm[0] = 1/ttot**2*2*ttot/T*numpy.absolute(vx_fft_norm[0])**2
					powerx_norm[len_pvec-1] = 1/ttot**2*2*ttot/T*numpy.absolute(vx_fft_norm[len_pvec-1])**2
							
					for k in range(1,len_pvec-1):
						powerx_norm[k] = 1/ttot**2*ttot/T*(numpy.absolute(vx_fft_norm[k])**2 + numpy.absolute(vx_fft_norm[ttot-k])**2)
					
					powery_norm = numpy.zeros((len_pvec,),dtype='float')
							
					powery_norm[0] = 1/ttot**2*2*ttot/T*numpy.absolute(vy_fft_norm[0])**2
					powery_norm[len_pvec-1] = 1/ttot**2*2*ttot/T*numpy.absolute(vy_fft_norm[len_pvec-1])**2
							
					for k in range(1,len_pvec-1):
						powery_norm[k] = 1/ttot**2*ttot/T*(numpy.absolute(vy_fft_norm[k])**2 + numpy.absolute(vy_fft_norm[ttot-k])**2)
					
					power_norm_list.append(powerx_norm+powery_norm)
					####
					
			power_norm_array = numpy.array(power_norm_list)
			
			power_array = numpy.array(power_list)
			norm_factor_array = numpy.array(norm_factor_list)
			
			psd_overall = numpy.sum(power_array, axis=0)/numpy.sum(norm_factor_array)
			psd_overall_normed = numpy.sum(power_norm_array, axis=0)/numpy.sum(norm_factor_array)
			nx,ny = power_array.shape
			#psd_overall_normed = numpy.sum(power_array/numpy.tile(numpy.array(var_list),(ny,1)).T, axis=0)/numpy.sum(norm_factor_array)
			
			log_freqs = numpy.log10(hz_frequencies[1:]*60)
			
			#####Piecewise linear fits
			fit_params_overall_pl = curve_fit(piecewise_linear,log_freqs,numpy.log10(psd_overall[1:]),p0=[-1,-1.5,0,-.5])[0]
			fit_params_overall_normed_pl = curve_fit(piecewise_linear,log_freqs,numpy.log10(psd_overall_normed[1:]),p0=[-1,-1.5,0,-.5])[0]
			print(fit_params_overall_normed_pl)

			#fit_params_lorentz_normed = curve_fit(Lorentzian_with_noise,hz_frequencies[1:]*60,psd_overall_normed[1:])[0]
			#fit_params_lorentz_overall = curve_fit(Lorentzian_with_noise,hz_frequencies[1:]*60,numpy.log10(psd_overall[1:]))[0]
			print('PSD calc ntraj, ',len(power_list))
			ax.loglog(hz_frequencies[1:]*60, psd_overall[1:],'o',markersize=3,alpha=.8)
			#fit1,=ax.loglog(hz_frequencies[1:]*60, numpy.power(10,piecewise_linear(log_freqs, *fit_params_overall_pl)), 'k')
			#fit1,=ax.loglog(hz_frequencies[1:]*60, Lorentzian_with_noise(hz_frequencies[1:]*60, *fit_params_lorentz_overall), 'k')
			fit1,=ax.loglog(hz_frequencies[1:20]*60, numpy.power(10,1.05*fit_params_overall_pl[1]*numpy.ones(len(hz_frequencies[1:20],))), 'k')
			fit2,=ax.loglog(hz_frequencies[10:]*60, numpy.power(10,linear_part2(log_freqs[9:], *fit_params_overall_pl)), 'k--')
			ax.set_ylim(2*10**-2,.4)
			#ax.loglog(hz_frequencies[1:]*60, numpy.power(10,Lorentzian_with_noise(hz_frequencies[1:]*60, *fit_params_lorentz_overall)), 'k')
			ax.set_xlabel('Frequency (1/min)')
			ax.legend([fit1,fit2],['Simple random walk','Levy flight'],loc="lower left",fontsize=9)
			ax.set_ylabel(r'$\langle PSD(f) \rangle$')
			#inset_ax.loglog(hz_frequencies[1:]*60, psd_overall_normed[1:],'o',markersize=3,alpha=.8)
			#inset_ax.loglog(hz_frequencies[1:]*60, numpy.power(10,piecewise_linear(log_freqs, *fit_params_overall_normed_pl)), 'k')
			#inset_ax.text(1,.1,'Normalized',horizontalalignment='center')

def PSDs_overall_np(corr_times_dict, trajectory_dict_polar, ax):
	
	experiment_list = ['fish_T']
	experiment_titles = ['D. rerio','M. musculus','Dictyostelium']

	study_labels = ['This study','Gerard et al.','Dang et al.']
	treatment_label_dict = {'control':'Control','rockout':'12 $\mu$M Rockout','Myo1g_KO':'Myo1g KO','Arp_KO':'Arpin KO','Arp_rescue':'Arpin rescue','WT':'WT'}
	experiment_ind = 0

	tmax = 100
	traj_list = []
	mean_speed_list = []

	ttot = 100
	hz_frequencies = 2*numpy.pi*(1/45)*numpy.arange(int(ttot/2))*1/ttot

	for experiment in experiment_list:
		
		treatment_list = []
		treatment_label_list = []
		for treatment in corr_times_dict[experiment]:
			if 'control' in treatment and 'highfreq' not in treatment:
				for sample in trajectory_dict_polar[experiment][treatment]:
					
					for traj_ind in trajectory_dict_polar[experiment][treatment][sample]:
						
						traj_data = numpy.array(trajectory_dict_polar[experiment][treatment][sample][traj_ind]).T
						
						if (traj_data[-1,0] - traj_data[0,0])/45 > tmax and numpy.sum(numpy.isnan(traj_data)) < .5:
							traj_list.append((sample,traj_ind))
							mean_speed_list.append(numpy.mean(traj_data[:,1]))
			
				len_pvec = int(ttot/2)
				
				
				power_list = []
				power_norm_list = []
				powerx_list = []
				norm_factor_list = []
				var_list = []
				for traj_tuple in traj_list:
					
					sample = traj_tuple[0]
					traj_ind = traj_tuple[1]
					
					interval = .75
					
					#interval_freqs = hz_frequencies*interval
				
					traj_data = numpy.array(trajectory_dict_polar[experiment][treatment][sample][traj_ind]).T
					
					times = traj_data[:,0]/45
					vx = traj_data[:,1]*numpy.cos(traj_data[:,2])/interval
					vy = traj_data[:,1]*numpy.sin(traj_data[:,2])/interval
					
					vx_norm = numpy.cos(traj_data[:,2])
					vy_norm = numpy.sin(traj_data[:,2])
					
					T = int(times[-1]-times[0])
					
					times = (traj_data[:,0] - traj_data[0,0])/45 ####Start all trajectories at t=0; ft is invariant under a translation in time
							
							
					###in case there are missing timepoints, interpolate the speeds vector from tp0 to tptmax
						
					#temp_vx = interpolate.interp1d(times,vx)
					#temp_vy = interpolate.interp1d(times,vy)
							
					vx_vec = vx
					vy_vec = vy
							
					####Zero pad out to ttot timepoints for shorter trajectories. (Note that we will need to be careful with the normalization to preserve the correct amplitude)
							
					#if T < ttot:
					#	nadd = ttot - T
					#	vx_vec = numpy.concatenate((vx_vec, numpy.zeros((nadd,)) ))
					#	vy_vec = numpy.concatenate((vy_vec, numpy.zeros((nadd,)) ))
								
					vx_fft = numpy.fft.fft(vx_vec)
					vy_fft = numpy.fft.fft(vy_vec)	
					
					powerx = numpy.zeros((len_pvec,),dtype='float')
							
					powerx[0] = 1/ttot**2*2*numpy.absolute(vx_fft[0])**2
					powerx[len_pvec-1] = 1/ttot**2*2*numpy.absolute(vx_fft[len_pvec-1])**2
							
					for k in range(1,len_pvec-1):
						powerx[k] = 1/ttot**2*(numpy.absolute(vx_fft[k])**2 + numpy.absolute(vx_fft[ttot-k])**2)
					
					powery = numpy.zeros((len_pvec,),dtype='float')
							
					powery[0] = 1/ttot**2*2*numpy.absolute(vy_fft[0])**2
					powery[len_pvec-1] = 1/ttot**2*2*numpy.absolute(vy_fft[len_pvec-1])**2
							
					for k in range(1,len_pvec-1):
						powery[k] = 1/ttot**2*(numpy.absolute(vy_fft[k])**2 + numpy.absolute(vy_fft[ttot-k])**2)
					
					power_list.append(powerx+powery)
					norm_factor_list.append(ttot/T)
					
					####
					#if T < ttot:
					#	nadd = ttot - T
					#	vx_norm = numpy.concatenate((vx_norm, numpy.zeros((nadd,)) ))
					#	vy_norm = numpy.concatenate((vy_norm, numpy.zeros((nadd,)) ))
								
					vx_fft_norm = numpy.fft.fft(vx_norm)
					vy_fft_norm = numpy.fft.fft(vy_norm)	
						
					powerx_norm = numpy.zeros((len_pvec,),dtype='float')
							
					powerx_norm[0] = 1/ttot**2*2*numpy.absolute(vx_fft_norm[0])**2
					powerx_norm[len_pvec-1] = 1/ttot**2*2*numpy.absolute(vx_fft_norm[len_pvec-1])**2
							
					for k in range(1,len_pvec-1):
						powerx_norm[k] = 1/ttot**2*(numpy.absolute(vx_fft_norm[k])**2 + numpy.absolute(vx_fft_norm[ttot-k])**2)
					
					powery_norm = numpy.zeros((len_pvec,),dtype='float')
							
					powery_norm[0] = 1/ttot**2*2*numpy.absolute(vy_fft_norm[0])**2
					powery_norm[len_pvec-1] = 1/ttot**2*2*numpy.absolute(vy_fft_norm[len_pvec-1])**2
							
					for k in range(1,len_pvec-1):
						powery_norm[k] = 1/ttot**2*(numpy.absolute(vy_fft_norm[k])**2 + numpy.absolute(vy_fft_norm[ttot-k])**2)
					
					power_norm_list.append(powerx_norm+powery_norm)
					####
					
			power_norm_array = numpy.array(power_norm_list)
			
			power_array = numpy.array(power_list)
			norm_factor_array = numpy.array(norm_factor_list)
			
			psd_overall = numpy.sum(power_array, axis=0)/numpy.sum(norm_factor_array)
			psd_overall_normed = numpy.sum(power_norm_array, axis=0)/numpy.sum(norm_factor_array)
			nx,ny = power_array.shape
			#psd_overall_normed = numpy.sum(power_array/numpy.tile(numpy.array(var_list),(ny,1)).T, axis=0)/numpy.sum(norm_factor_array)
			
			log_freqs = numpy.log10(hz_frequencies[1:]*60)
			
			#####Piecewise linear fits
			fit_params_overall_pl = curve_fit(piecewise_linear,log_freqs,numpy.log10(psd_overall[1:]),p0=[-1,-1.5,0,-.5])[0]
			fit_params_overall_normed_pl = curve_fit(piecewise_linear,log_freqs,numpy.log10(psd_overall_normed[1:]),p0=[-1,-1.5,0,-.5])[0]

			#fit_params_lorentz_normed = curve_fit(Lorentzian_with_noise,hz_frequencies[1:]*60,psd_overall_normed[1:])[0]
			#fit_params_lorentz_overall = curve_fit(Lorentzian_with_noise,hz_frequencies[1:]*60,numpy.log10(psd_overall[1:]))[0]
			
			ax.loglog(hz_frequencies[1:]*60, psd_overall[1:],'o',markersize=3,alpha=.8)
			#fit1,=ax.loglog(hz_frequencies[1:]*60, numpy.power(10,piecewise_linear(log_freqs, *fit_params_overall_pl)), 'k')
			#fit1,=ax.loglog(hz_frequencies[1:]*60, Lorentzian_with_noise(hz_frequencies[1:]*60, *fit_params_lorentz_overall), 'k')
			#fit1,=ax.loglog(hz_frequencies[1:20]*60, numpy.power(10,1.05*fit_params_overall_pl[1]*numpy.ones(len(hz_frequencies[1:20],))), 'k')
			#fit2,=ax.loglog(hz_frequencies[10:]*60, numpy.power(10,linear_part2(log_freqs[9:], *fit_params_overall_pl)), 'k--')
			#ax.set_ylim(2*10**-2,.4)
			#ax.loglog(hz_frequencies[1:]*60, numpy.power(10,Lorentzian_with_noise(hz_frequencies[1:]*60, *fit_params_lorentz_overall)), 'k')
			ax.set_xlabel('Frequency (1/min)')
			#ax.legend([fit1,fit2],['Simple random walk','Levy flight'],loc="lower left",fontsize=9)
			ax.set_ylabel(r'$\langle PSD(f) \rangle$')
			#inset_ax.loglog(hz_frequencies[1:]*60, psd_overall_normed[1:],'o',markersize=3,alpha=.8)
			#inset_ax.loglog(hz_frequencies[1:]*60, numpy.power(10,piecewise_linear(log_freqs, *fit_params_overall_normed_pl)), 'k')
			#inset_ax.text(1,.1,'Normalized',horizontalalignment='center')		

def PSDs_by_speed_class(speeds_dict, trajectory_dict_polar, ax):
	
	experiment_list = ['fish_T']
	experiment_titles = ['D. rerio','M. musculus','Dictyostelium']

	study_labels = ['This study','Gerard et al.','Dang et al.']
	treatment_label_dict = {'control':'Control','rockout':'12 $\mu$M Rockout','Myo1g_KO':'Myo1g KO','Arp_KO':'Arpin KO','Arp_rescue':'Arpin rescue','WT':'WT'}
	experiment_ind = 0

	tmin = 50
	traj_list = {}
	mean_speed_list = {}

	ttot = 400
	hz_frequencies = 2*numpy.pi*(1/45)*numpy.arange(int(ttot/2))*1/ttot
	
	###Collect mean speeds to compute quintile speed classes
	
	mean_speed_list_all = []
	
	n_tsteps_min = 20
	
	for experiment in trajectory_dict_polar:
		if 'fish' in experiment:
			for treatment in trajectory_dict_polar[experiment]:
				if 'control' in treatment and 'highfreq' not in treatment:
					
					for sample in trajectory_dict_polar[experiment][treatment]:
						
						for traj_ind in trajectory_dict_polar[experiment][treatment][sample]:
							
							if traj_ind in speeds_dict[experiment][treatment][sample]:
								mean_speed=numpy.mean(speeds_dict[experiment][treatment][sample][traj_ind])
								if ~numpy.isnan(mean_speed) and len(speeds_dict[experiment][treatment][sample][traj_ind]) >= n_tsteps_min:
									mean_speed_list_all.append(mean_speed)
	
	speed_classes = numpy.percentile(mean_speed_list_all,numpy.arange(0,101,20))
	
	handles = []
	
	for experiment in experiment_list:
		
		treatment_list = []
		for treatment in trajectory_dict_polar[experiment]:
			if 'control' in treatment and 'highfreq' not in treatment:
				for sample in trajectory_dict_polar[experiment][treatment]:
					
					for traj_ind in trajectory_dict_polar[experiment][treatment][sample]:
						
						traj_data = numpy.array(trajectory_dict_polar[experiment][treatment][sample][traj_ind]).T
						mean_speed=numpy.mean(speeds_dict[experiment][treatment][sample][traj_ind])
						if (traj_data[-1,0] - traj_data[0,0])/45 > tmin and ~numpy.isnan(mean_speed) and numpy.sum(numpy.isnan(traj_data)) < .5:
						
							speed_class = numpy.argmax( mean_speed - speed_classes < 0 ) - 1
									
							if speed_class < 0:
								speed_class = len(speed_classes) - 1
							
							if speed_class not in traj_list:
								traj_list[speed_class] = []
								mean_speed_list[speed_class] = []
								
							traj_list[speed_class].append((sample,traj_ind))
							mean_speed_list[speed_class].append(mean_speed)
							
				len_pvec = int(ttot/2)
				
				speed_label_list = []
				
				for speed_class in range(5):
				
					powerx_list = []
					norm_factor_list = []
					var_list = []
					speed_label_list.append( str(numpy.round(numpy.mean(mean_speed_list[speed_class]),1)) + r' $\frac{\mu m}{min}$')
					
					for traj_tuple in traj_list[speed_class]:
						
						sample = traj_tuple[0]
						traj_ind = traj_tuple[1]
						
						interval = .75
						
						#interval_freqs = hz_frequencies*interval
					
						traj_data = numpy.array(trajectory_dict_polar[experiment][treatment][sample][traj_ind]).T
						
						vx = traj_data[:,1]*numpy.cos(traj_data[:,2])/interval
						vy = traj_data[:,1]*numpy.sin(traj_data[:,2])/interval
						
						#vx = numpy.cos(traj_data[:,2])
						#vy = numpy.sin(traj_data[:,2])
						
						times = (traj_data[:,0] - traj_data[0,0])/45 ####Start all trajectories at t=0; ft is invariant under a translation in time
						T = int(times[-1]-times[0])		
								
						
						vx_vec = vx
						vy_vec = vy
						
						####Zero pad out to ttot timepoints for shorter trajectories. (Note that we will need to be careful with the normalization to preserve the correct amplitude)
								
						if T < ttot:
							nadd = ttot - T
							vx_vec = numpy.concatenate((vx_vec, numpy.zeros((nadd,)) ))
							vy_vec = numpy.concatenate((vy_vec, numpy.zeros((nadd,)) ))
									
						vx_fft = numpy.fft.fft(vx_vec)
						vy_fft = numpy.fft.fft(vy_vec)	
							
						powerx = numpy.zeros((len_pvec,),dtype='float')
						
						#print(vx_vec)
						#print(vx_fft)
								
						powerx[0] = 1/ttot**2*2*ttot/T*numpy.absolute(vx_fft[0])**2
						powerx[len_pvec-1] = 1/ttot**2*2*ttot/T*numpy.absolute(vx_fft[len_pvec-1])**2
								
						for k in range(1,len_pvec-1):
							powerx[k] = 1/ttot**2*ttot/T*(numpy.absolute(vx_fft[k])**2 + numpy.absolute(vx_fft[ttot-k])**2)
						
						powery = numpy.zeros((len_pvec,),dtype='float')
								
						powery[0] = 1/ttot**2*2*ttot/T*numpy.absolute(vy_fft[0])**2
						powery[len_pvec-1] = 1/ttot**2*2*ttot/T*numpy.absolute(vy_fft[len_pvec-1])**2
								
						for k in range(1,len_pvec-1):
							powery[k] = 1/ttot**2*ttot/T*(numpy.absolute(vy_fft[k])**2 + numpy.absolute(vy_fft[ttot-k])**2)
						
						if numpy.sum(numpy.isnan(powerx)) > .5:
							print('why are there nans??', vx_vec)
						powerx_list.append(powerx)
						norm_factor_list.append(ttot/T)
						var_list.append(numpy.var(vx_vec))
						
					powerx_array = numpy.array(powerx_list)
					norm_factor_array = numpy.array(norm_factor_list)
					
					psd_overall = numpy.sum(powerx_array, axis=0)/numpy.sum(norm_factor_array)
					nx,ny = powerx_array.shape
				
				
					log_freqs = numpy.log10(hz_frequencies[1:]*60)
				
				
					fit_params_lorentz_overall = curve_fit(Lorentzian_with_noise,hz_frequencies[1:]*60,psd_overall[1:])[0]
					print(speed_class,fit_params_lorentz_overall)
					handle,=ax.loglog(hz_frequencies[1:]*60, psd_overall[1:],'o',markersize=3,alpha=.8)
					fit1,=ax.loglog(hz_frequencies[1:]*60, Lorentzian_with_noise(hz_frequencies[1:]*60, *fit_params_lorentz_overall), 'k')
				
					ax.set_xlabel('Frequency (1/min)')
				
					handles.append(handle)
	lgnd = ax.legend(handles,speed_label_list,fontsize=8,loc="upper right",handletextpad=.2,frameon=False)#markerscale=1,
	
	ax.set_ylabel(r'$\langle \tilde{v}_x^2(f) \rangle$')
	ax.set_xlim(10**-2,20)
	ax.set_ylim(5*10**-3,1)
def MSDs_by_speed_class_fishonly(speeds_dict, step_angle_coupling_dict, drift_corrected_traj_dict, selectable_inds_by_speed_class, ax):
	
	msds_by_tau = {}
	mean_speeds = {}
	
	taus = [45,90,135,180,270,360,450,540,630,720,810,900,990,1080]
	
	msds_by_tau = {}
	for experiment in speeds_dict:
		for treatment in speeds_dict[experiment]:
			if 'control' in treatment:	
				for sample in speeds_dict[experiment][treatment]:
					for tau in taus:
						
						msds_by_traj = trajectory_analysis_functions_09192019.calculate_MSDs(drift_corrected_traj_dict[experiment][treatment][sample], tau, overlap=True)
							
						for traj_ind in msds_by_traj:
							full_ind = sample + '-' + traj_ind
							included = False
							for sc in range(5):
								if full_ind in set(selectable_inds_by_speed_class[sc]):
									included = True
									speed_class = sc
									break
							if included:
								mean_speed=numpy.mean(speeds_dict[experiment][treatment][sample][traj_ind])
									
								if speed_class not in mean_speeds:
									mean_speeds[speed_class] = []
								
								mean_speeds[speed_class].append( mean_speed	)
									
								if speed_class not in msds_by_tau:
									msds_by_tau[speed_class] = {}
								if tau not in msds_by_tau[speed_class]:
									msds_by_tau[speed_class][tau] = {}
								
								msds_by_tau[speed_class][tau][full_ind] = msds_by_traj[traj_ind]
									
			
	n_bootstrap = 200
	speed_label_list = []
	for speed_class in range(5):
		indexes = selectable_inds_by_speed_class[speed_class]
		n_cells = len(indexes)
		msds_bs = []
		speed_label_list.append( 'S = ' + str(numpy.round(numpy.mean(mean_speeds[speed_class]),1)) + r' $\frac{\mu m}{min}$')
		for n in range(n_bootstrap):
			
			cell_choices = numpy.random.randint(0,n_cells,size=n_cells)
			
			index_choices = indexes[cell_choices]
			
			mean_msd_of_tau = []
			for tau in taus:
				
				msd = 0
				interval_counter = 0
				for index in index_choices:
					msd += numpy.sum( msds_by_tau[speed_class][tau][index] )
					interval_counter += len( msds_by_tau[speed_class][tau][index] )
				msd = msd/interval_counter
				
				mean_msd_of_tau.append(msd)
			
			msds_bs.append(mean_msd_of_tau)
		
		msds_bs = numpy.array(msds_bs)
		
		msd_med = numpy.percentile(msds_bs,50,axis=0)
		lb = numpy.percentile(msds_bs,5,axis=0)
		ub = numpy.percentile(msds_bs,95,axis=0)
		
		msd_med_scaled = msd_med/msd_med[0]
		lb_scaled = lb/msd_med[0]
		ub_scaled = ub/msd_med[0]
		
		ax.set_xscale('log')
		ax.set_yscale('log')
		
		ax.errorbar(numpy.array(taus)/60,msd_med_scaled,yerr=[msd_med_scaled-lb_scaled,ub_scaled-msd_med_scaled],marker='o',markersize=3,elinewidth=.5)
		ax.set_xlabel(r'$\tau (min)$')
		ax.set_ylabel(r'$\frac{\langle MSD(\tau) \rangle}{\langle MSD_0 \rangle}$')
		lgnd = ax.legend(speed_label_list,fontsize=8,ncol=1,loc='upper left',frameon=False)

def MSDs_by_speed_class_fishonly_scaled(speeds_dict, step_angle_coupling_dict, drift_corrected_traj_dict, selectable_inds_by_speed_class, ax):
	
	msds_by_tau = {}
	mean_speeds = {}
	
	taus = [45,90,135,180,270,360,450,540,630,720,810,900,990,1080]
	alpha = .15
	beta = .1
	msds_by_tau = {}
	for experiment in speeds_dict:
		for treatment in speeds_dict[experiment]:
			if 'control' in treatment:	
				for sample in speeds_dict[experiment][treatment]:
					for tau in taus:
						
						msds_by_traj = trajectory_analysis_functions_09192019.calculate_MSDs(drift_corrected_traj_dict[experiment][treatment][sample], tau, overlap=True)
							
						for traj_ind in msds_by_traj:
							full_ind = sample + '-' + traj_ind
							included = False
							for sc in range(5):
								if full_ind in set(selectable_inds_by_speed_class[sc]):
									included = True
									speed_class = sc
									break
							if included:
								mean_speed=numpy.mean(speeds_dict[experiment][treatment][sample][traj_ind])
									
								if speed_class not in mean_speeds:
									mean_speeds[speed_class] = []
								
								mean_speeds[speed_class].append( mean_speed	)
									
								if speed_class not in msds_by_tau:
									msds_by_tau[speed_class] = {}
								if tau not in msds_by_tau[speed_class]:
									msds_by_tau[speed_class][tau] = {}
								
								msds_by_tau[speed_class][tau][full_ind] = msds_by_traj[traj_ind]/(mean_speed**2*(alpha*mean_speed + beta)**2)
									
			
	n_bootstrap = 200
	speed_label_list = []
	for speed_class in range(5):
		indexes = selectable_inds_by_speed_class[speed_class]
		n_cells = len(indexes)
		msds_bs = []
		speed_label_list.append( 'S = ' + str(numpy.round(numpy.mean(mean_speeds[speed_class]),1)) + r' $\frac{\mu m}{min}$')
		for n in range(n_bootstrap):
			
			cell_choices = numpy.random.randint(0,n_cells,size=n_cells)
			
			index_choices = indexes[cell_choices]
			
			mean_msd_of_tau = []
			for tau in taus:
				
				msd = 0
				interval_counter = 0
				for index in index_choices:
					msd += numpy.sum( msds_by_tau[speed_class][tau][index] )
					interval_counter += len( msds_by_tau[speed_class][tau][index] )
				msd = msd/interval_counter
				
				mean_msd_of_tau.append(msd)
			
			msds_bs.append(mean_msd_of_tau)
		
		msds_bs = numpy.array(msds_bs)
		
		msd_med = numpy.percentile(msds_bs,50,axis=0)
		lb = numpy.percentile(msds_bs,5,axis=0)
		ub = numpy.percentile(msds_bs,95,axis=0)
		
		msd_med_scaled = msd_med#/msd_med[0]
		lb_scaled = lb#/msd_med[0]
		ub_scaled = ub#/msd_med[0]
		
		ax.set_xscale('log')
		ax.set_yscale('log')
		S = numpy.mean(mean_speeds[speed_class])
		alpha = .4
		beta = .1
		ax.errorbar(numpy.array(taus)/60/(S*alpha + beta),msd_med_scaled,yerr=[msd_med_scaled-lb_scaled,ub_scaled-msd_med_scaled],marker='o',markersize=3,elinewidth=.5)
		ax.set_xlabel(r'$\tau (min)$')
		ax.set_ylabel(r'$\frac{\langle MSD(\tau) \rangle}{\langle MSD_0 \rangle}$')
		lgnd = ax.legend(speed_label_list,fontsize=8,ncol=1,loc='upper left',frameon=False)

def MSDs_by_speed_class_fishonly_sub(speeds_dict, step_angle_coupling_dict, drift_corrected_traj_dict, selectable_inds_by_speed_class, ax):
	
	msds_by_tau = {}
	mean_speeds = {}
	
	taus = [90,180,270,360,450,540,630,720,810,900,990,1080]
	
	msds_by_tau = {}
	for experiment in speeds_dict:
		for treatment in speeds_dict[experiment]:
			if 'control' in treatment:	
				for sample in speeds_dict[experiment][treatment]:
					for tau in taus:
						
						msds_by_traj = trajectory_analysis_functions_09192019.calculate_MSDs(drift_corrected_traj_dict[experiment][treatment][sample], tau, overlap=True)
							
						for traj_ind in msds_by_traj:
							full_ind = sample + '-' + traj_ind
							included = False
							for sc in range(5):
								if full_ind in set(selectable_inds_by_speed_class[sc]):
									included = True
									speed_class = sc
									break
							if included:
								mean_speed=numpy.mean(speeds_dict[experiment][treatment][sample][traj_ind])
									
								if speed_class not in mean_speeds:
									mean_speeds[speed_class] = []
								
								mean_speeds[speed_class].append( mean_speed	)
									
								if speed_class not in msds_by_tau:
									msds_by_tau[speed_class] = {}
								if tau not in msds_by_tau[speed_class]:
									msds_by_tau[speed_class][tau] = {}
								
								msds_by_tau[speed_class][tau][full_ind] = msds_by_traj[traj_ind]
									
			
	n_bootstrap = 200
	speed_label_list = []
	for speed_class in range(5):
		indexes = selectable_inds_by_speed_class[speed_class]
		n_cells = len(indexes)
		msds_bs = []
		speed_label_list.append( 'S = ' + str(numpy.round(numpy.mean(mean_speeds[speed_class]),1)) + r' $\frac{\mu m}{min}$')
		for n in range(n_bootstrap):
			
			cell_choices = numpy.random.randint(0,n_cells,size=n_cells)
			
			index_choices = indexes[cell_choices]
			
			mean_msd_of_tau = []
			for tau in taus:
				
				msd = 0
				interval_counter = 0
				for index in index_choices:
					msd += numpy.sum( msds_by_tau[speed_class][tau][index] )
					interval_counter += len( msds_by_tau[speed_class][tau][index] )
				msd = msd/interval_counter
				
				mean_msd_of_tau.append(msd)
			
			msds_bs.append(mean_msd_of_tau)
		
		msds_bs = numpy.array(msds_bs)
		
		msd_med = numpy.percentile(msds_bs,50,axis=0)
		lb = numpy.percentile(msds_bs,5,axis=0)
		ub = numpy.percentile(msds_bs,95,axis=0)
		
		msd_med_scaled = msd_med/msd_med[0]
		lb_scaled = lb/msd_med[0]
		ub_scaled = ub/msd_med[0]
		
		ax.set_xscale('log')
		ax.set_yscale('log')
		
		ax.errorbar(numpy.array(taus)/60,msd_med_scaled,yerr=[msd_med_scaled-lb_scaled,ub_scaled-msd_med_scaled],marker='o',markersize=3,elinewidth=.5)
		ax.set_xlabel(r'$\tau (min)$')
		ax.set_ylabel(r'$\frac{\langle MSD(\tau) \rangle}{\langle MSD_0 \rangle}$')
		lgnd = ax.legend(speed_label_list,fontsize=8,ncol=1,loc='upper left',frameon=False)

def speed_rank_corr(trajectory_dict_polar, speeds_dict, ax):
	
	def calculate_rankcorr(mspeed_arr):
	
		rho,p = spearmanr(mspeed_arr,axis=0)
		avg_vals = []
	
		for i in range(rho.shape[0]-1):
			avg_vals.append( numpy.mean( numpy.diagonal(rho,offset=i+1) ) )
		return avg_vals
	

	#####Focusing on the 20 min interval; plot rank correlation (bootstrapped over cells) as well as expectation based on random permutation.
	#####Also plot the correlation matrix

	all_trajs_chunked_speeds = []

	n = 26
	tint = 45*n
	time_chunks4 = numpy.arange(tint-45,6000+tint,tint)
	for experiment in trajectory_dict_polar:
			if 'fish' in experiment:
				for treatment in trajectory_dict_polar[experiment]:
					if 'control' in treatment and 'highfreq' not in treatment:
						
						for sample in trajectory_dict_polar[experiment][treatment]:
							
							for traj_ind in trajectory_dict_polar[experiment][treatment][sample]:
								
								if traj_ind in speeds_dict[experiment][treatment][sample] and trajectory_dict_polar[experiment][treatment][sample][traj_ind][0][-1]- trajectory_dict_polar[experiment][treatment][sample][traj_ind][0][0]> time_chunks4[-1]:
									traj = trajectory_dict_polar[experiment][treatment][sample][traj_ind]
									timesteps = traj[0]
									speeds = []
									sloc = 0
									tc = 0
									counter = 0
									s = 0
									while sloc < len(timesteps) and tc < len(time_chunks4):
										s += traj[1][sloc]
										counter += 1.
										if timesteps[sloc]-timesteps[0] > time_chunks4[tc]:
											speeds.append(s/counter)
											tc += 1
											s = 0
											counter = 0
										sloc += 1
									if numpy.sum(numpy.isnan(numpy.array(speeds))>.5):
										print(traj_ind,len(speeds))
									all_trajs_chunked_speeds.append(speeds)

	speed_arr4 = numpy.array(all_trajs_chunked_speeds)
	print('ntraj speed corr, ',len(all_trajs_chunked_speeds))
	n_bootstrap = 500

	corr_vals_bs = []
	####Bootstrap over cells
	ncells = speed_arr4.shape[0]
	for n in range(n_bootstrap):
		cell_choices = numpy.random.choice(numpy.arange(ncells),size=ncells)
		speed_arr_bs = speed_arr4[cell_choices,:]
		corr_vals = calculate_rankcorr(speed_arr_bs)
		corr_vals_bs.append(corr_vals)

	corr_vals_bs = numpy.array(corr_vals_bs)
	corr_vals_50 = numpy.percentile(corr_vals_bs,50,axis=0)
	corr_vals_95 = numpy.percentile(corr_vals_bs,95,axis=0)
	corr_vals_5 = numpy.percentile(corr_vals_bs,5,axis=0)

	####Calculate correlation values under a random permutation of each column, as a sanity check

	null_bs = []
	for n in range(n_bootstrap):
		speed_arr_bs = numpy.zeros_like(speed_arr4)
		for j in range(speed_arr4.shape[1]):
			speed_arr_bs[:,j] = numpy.random.permutation(speed_arr4[:,j])
		
		null_vals = calculate_rankcorr(speed_arr_bs)
		null_bs.append(null_vals)

	null_vals_bs = numpy.array(null_bs)
	null_vals_50 = numpy.percentile(null_vals_bs,50,axis=0)
	null_vals_95 = numpy.percentile(null_vals_bs,95,axis=0)
	null_vals_5 = numpy.percentile(null_vals_bs,5,axis=0)

	ax.errorbar((time_chunks4[:-1]+45)/60., corr_vals_50, yerr=[corr_vals_50-corr_vals_5,corr_vals_95-corr_vals_50], marker='o',markersize=4)
	ax.errorbar((time_chunks4[:-1]+45)/60., null_vals_50, yerr=[null_vals_50-null_vals_5,null_vals_95-null_vals_50], marker='o',markersize=4)
	ax.set_xlabel('Time (min)')
	ax.set_ylabel(r'Speed rank correlation ($\rho$)')
	ax.legend(['Trajectories','Null (permuted)'])


###Note: next fcn not currently in use		
def MSD_dists_by_speed_class(speeds_dict, drift_corrected_traj_dict, tau, ax):
	
	msds_by_tau = {}
	mean_speeds = {}
	indexes = []
	
	taus = [tau]#,1080+180*1,1080+180*2,1080+180*3,1080+180*4,1080+180*6,1080+180*8]#,1080+180*10,1080+180*14,1080+180*18,1080+180*22,1080+180*24,1080+180*30]#numpy.arange(0,45*100,45)
	
	###Collect mean speeds to compute quintile speed classes
	n_tsteps_min = 30
	
	mean_speed_list_all = []
	
	for experiment in drift_corrected_traj_dict:
		if 'fish' in experiment:
			for treatment in drift_corrected_traj_dict[experiment]:
				if 'control' in treatment and 'highfreq' not in treatment:
					
					for sample in drift_corrected_traj_dict[experiment][treatment]:
						
						for traj_ind in drift_corrected_traj_dict[experiment][treatment][sample]:
							
							if traj_ind in speeds_dict[experiment][treatment][sample]:
								mean_speed=numpy.mean(speeds_dict[experiment][treatment][sample][traj_ind])
								if ~numpy.isnan(mean_speed) and len(speeds_dict[experiment][treatment][sample][traj_ind]) >= n_tsteps_min:
									mean_speed_list_all.append(mean_speed)
	
	speed_classes = numpy.percentile(mean_speed_list_all,numpy.arange(0,101,20))
	print(speed_classes)
	for experiment in speeds_dict:
		if 'fish' in experiment:
			for treatment in speeds_dict[experiment]:
				if 'control' in treatment and 'highfreq' not in treatment:
					for sample in speeds_dict[experiment][treatment]:
						for tau in taus:
						
							msds_by_traj = trajectory_analysis_functions_09192019.calculate_MSDs(drift_corrected_traj_dict[experiment][treatment][sample], tau, overlap=True)
							
							for traj_ind in msds_by_traj:
								
								if traj_ind in speeds_dict[experiment][treatment][sample]:
									mean_speed=numpy.mean(speeds_dict[experiment][treatment][sample][traj_ind])
									
									if ~numpy.isnan(mean_speed):
										speed_class = numpy.argmax( mean_speed - speed_classes < 0 ) - 1
									
										if speed_class < 0:
											speed_class = len(speed_classes) - 1
										
										full_ind = sample + '-' + traj_ind
										if speed_class not in msds_by_tau:
											msds_by_tau[speed_class] = {}
										if speed_class not in mean_speeds:
											mean_speeds[speed_class] = {}
										if tau not in msds_by_tau[speed_class]:
											msds_by_tau[speed_class][tau] = {}
											
										msds_by_tau[speed_class][tau][full_ind] = msds_by_traj[traj_ind]
										
										if full_ind not in mean_speeds[speed_class]:
											mean_speeds[speed_class][full_ind] = mean_speed
	
	selectable_inds_by_speed_class = {}
	
	for speed_class in range(5):
		
		inds0 = set(msds_by_tau[speed_class][taus[0]].keys())
		
		for tau in taus[1:]:
			inds0.intersection_update(set(msds_by_tau[speed_class][tau].keys()))
		#print(inds0)
		selectable_inds_by_speed_class[speed_class] = numpy.array(list(inds0))
		#print(selectable_inds_by_speed_class[speed_class],speed_class)
		
	mean_speed_restricted = {}
	for speed_class in range(5):
		mean_speed_restricted[speed_class] = []
		#print(speed_class)
		for index in selectable_inds_by_speed_class[speed_class]:
			mean_speed_restricted[speed_class].append(mean_speeds[speed_class][index])
			
	n_bootstrap = 200
	speed_label_list = []
	for tau in taus:
		for speed_class in range(5):
			indexes = selectable_inds_by_speed_class[speed_class]
			n_cells = len(indexes)
			msds_bs = []
			speed_label_list.append( 'S = ' + str(numpy.round(numpy.mean(mean_speed_restricted[speed_class]),1)) + r' $\frac{\mu m}{min}$')
			
			###Calculate bins and statistic over all cells
			msd_list = []
			for index in indexes:
				msd_list.extend( numpy.sqrt(numpy.array(msds_by_tau[speed_class][tau][index]))/mean_speeds[speed_class][index] )
			
			msd_bins = numpy.percentile(msd_list,numpy.arange(0,101,10))
			
			msd_stat,binedges = numpy.histogram(msd_list,bins=msd_bins,density=True)
			xlocs,binedges,nbins = binned_statistic(msd_list,msd_list,bins=msd_bins)
			
			###Bootstrap for error bars on the distribution
			msd_stat_bs = []
			for n in range(n_bootstrap):
				
				cell_choices = numpy.random.randint(0,n_cells,size=n_cells)
				
				index_choices = indexes[cell_choices]
				
				msd_list = []
				for index in index_choices:
					msd_list.extend( numpy.sqrt(numpy.array(msds_by_tau[speed_class][tau][index]))/mean_speeds[speed_class][index] )
				
				msd_dist,binedges = numpy.histogram(msd_list, bins=msd_bins,density=True)
				msd_stat_bs.append(msd_dist)
			
			msd_stat_bs = numpy.array(msd_stat_bs)
			medians = numpy.percentile(msd_stat_bs,50,axis=0)
			lb = numpy.percentile(msd_stat_bs,5,axis=0)
			ub = numpy.percentile(msd_stat_bs,95,axis=0)
			
			print(medians,msd_stat)
			
			ax.set_yscale('log')
			
			ax.errorbar(xlocs,msd_stat,yerr=[msd_stat - lb,ub-msd_stat],marker='o',markersize=3,elinewidth=.5)
		ax.set_xlabel(r'$\Delta x(\tau)/S$')
		ax.set_ylabel('Probability density')
##Note: next fcn not currently in use
def MSD_dists_by_speed_class_scale2(speeds_dict, drift_corrected_traj_dict, tau, ax):
	
	msds_by_tau = {}
	mean_speeds = {}
	indexes = []
	
	taus = [tau]#,1080+180*1,1080+180*2,1080+180*3,1080+180*4,1080+180*6,1080+180*8]#,1080+180*10,1080+180*14,1080+180*18,1080+180*22,1080+180*24,1080+180*30]#numpy.arange(0,45*100,45)
	
	###Collect mean speeds to compute quintile speed classes
	n_tsteps_min = 30
	
	mean_speed_list_all = []
	
	for experiment in drift_corrected_traj_dict:
		if 'fish' in experiment:
			for treatment in drift_corrected_traj_dict[experiment]:
				if 'control' in treatment and 'highfreq' not in treatment:
					
					for sample in drift_corrected_traj_dict[experiment][treatment]:
						
						for traj_ind in drift_corrected_traj_dict[experiment][treatment][sample]:
							
							if traj_ind in speeds_dict[experiment][treatment][sample]:
								mean_speed=numpy.mean(speeds_dict[experiment][treatment][sample][traj_ind])
								if ~numpy.isnan(mean_speed) and len(speeds_dict[experiment][treatment][sample][traj_ind]) >= n_tsteps_min:
									mean_speed_list_all.append(mean_speed)
	
	speed_classes = numpy.percentile(mean_speed_list_all,numpy.arange(0,101,20))
	print(speed_classes)
	for experiment in speeds_dict:
		if 'fish' in experiment:
			for treatment in speeds_dict[experiment]:
				if 'control' in treatment and 'highfreq' not in treatment:
					for sample in speeds_dict[experiment][treatment]:
						for tau in taus:
						
							msds_by_traj = trajectory_analysis_functions_09192019.calculate_MSDs(drift_corrected_traj_dict[experiment][treatment][sample], tau, overlap=True)
							
							for traj_ind in msds_by_traj:
								
								if traj_ind in speeds_dict[experiment][treatment][sample]:
									mean_speed=numpy.mean(speeds_dict[experiment][treatment][sample][traj_ind])
									
									if ~numpy.isnan(mean_speed):
										speed_class = numpy.argmax( mean_speed - speed_classes < 0 ) - 1
									
										if speed_class < 0:
											speed_class = len(speed_classes) - 1
										
										full_ind = sample + '-' + traj_ind
										if speed_class not in msds_by_tau:
											msds_by_tau[speed_class] = {}
										if speed_class not in mean_speeds:
											mean_speeds[speed_class] = {}
										if tau not in msds_by_tau[speed_class]:
											msds_by_tau[speed_class][tau] = {}
											
										msds_by_tau[speed_class][tau][full_ind] = msds_by_traj[traj_ind]
										
										if full_ind not in mean_speeds[speed_class]:
											mean_speeds[speed_class][full_ind] = mean_speed
	
	selectable_inds_by_speed_class = {}
	
	for speed_class in range(5):
		
		inds0 = set(msds_by_tau[speed_class][taus[0]].keys())
		
		for tau in taus[1:]:
			inds0.intersection_update(set(msds_by_tau[speed_class][tau].keys()))
		#print(inds0)
		selectable_inds_by_speed_class[speed_class] = numpy.array(list(inds0))
		#print(selectable_inds_by_speed_class[speed_class],speed_class)
		
	mean_speed_restricted = {}
	for speed_class in range(5):
		mean_speed_restricted[speed_class] = []
		#print(speed_class)
		for index in selectable_inds_by_speed_class[speed_class]:
			mean_speed_restricted[speed_class].append(mean_speeds[speed_class][index])
			
	n_bootstrap = 200
	speed_label_list = []
	for tau in taus:
		for speed_class in range(5):
			indexes = selectable_inds_by_speed_class[speed_class]
			n_cells = len(indexes)
			msds_bs = []
			speed_label_list.append( 'S = ' + str(numpy.round(numpy.mean(mean_speed_restricted[speed_class]),1)) + r' $\frac{\mu m}{min}$')
			
			###Calculate bins and statistic over all cells
			msd_list = []
			for index in indexes:
				msd_list.extend( numpy.sqrt(numpy.array(msds_by_tau[speed_class][tau][index]))/mean_speeds[speed_class][index]**1.5 )
			
			msd_bins = numpy.percentile(msd_list,numpy.arange(0,101,10))
			
			msd_stat,binedges = numpy.histogram(msd_list,bins=msd_bins,density=True)
			xlocs,binedges,nbins = binned_statistic(msd_list,msd_list,bins=msd_bins)
			
			###Bootstrap for error bars on the distribution
			msd_stat_bs = []
			for n in range(n_bootstrap):
				
				cell_choices = numpy.random.randint(0,n_cells,size=n_cells)
				
				index_choices = indexes[cell_choices]
				
				msd_list = []
				for index in index_choices:
					msd_list.extend( numpy.sqrt(numpy.array(msds_by_tau[speed_class][tau][index]))/mean_speeds[speed_class][index]**1.5 )
				
				msd_dist,binedges = numpy.histogram(msd_list, bins=msd_bins,density=True)
				msd_stat_bs.append(msd_dist)
			
			msd_stat_bs = numpy.array(msd_stat_bs)
			lb = numpy.percentile(msd_stat_bs,5,axis=0)
			ub = numpy.percentile(msd_stat_bs,95,axis=0)
			msd_stat = numpy.array(msd_stat)
			
			ax.set_yscale('log')
			
			ax.errorbar(xlocs,msd_stat,yerr=[msd_stat - lb,ub-msd_stat],marker='o',markersize=3,elinewidth=.5)
		ax.set_xlabel(r'$\Delta x(\tau)/S^{3/2}$')
		#ax.set_ylabel('Probability density')
		#lgnd = ax.legend(speed_label_list,fontsize=8,ncol=1,loc='upper left',frameon=False)
		
		


def MSD_v_speed(speeds_dict, corr_times_dict, drift_corrected_traj_dict, ax):
	
	
	def make_spc(a,b):
		def spc_fixed_ratio(x,c):
			return numpy.log10(c*x**2*(a*x + b))
		return spc_fixed_ratio
		
	def spc(x,a,b):
		return numpy.log10(a*x**2*(x + b/a))
	
	def prw(x,a):
		return numpy.log10(a*x**2)
	
	def linmod(x,a,b):
		return a*x + b
		
	msds_by_tau = {}
	mean_speeds = {}
	indexes = []
	
	corr_times_coll = {}
	speeds = []
	index_list = []
	
	corr_times_bootstrapped = []
	speeds_bootstrapped = []
	
	nbootstrap = 500
	
	for experiment in corr_times_dict:
		if 'fish' in experiment:
			for treatment in corr_times_dict[experiment]:
				if 'control' in treatment and 'highfreq' not in treatment:
					
					for sample in corr_times_dict[experiment][treatment]:
						
						for traj_ind in corr_times_dict[experiment][treatment][sample]:
							
							if traj_ind in speeds_dict[experiment][treatment][sample] and len(speeds_dict[experiment][treatment][sample][traj_ind]) > .5:
								
								mean_speed=numpy.mean(speeds_dict[experiment][treatment][sample][traj_ind])
									
								full_index = sample + '_' + traj_ind	
								corr_times_coll[full_index] = corr_times_dict[experiment][treatment][sample][traj_ind]
								index_list.append(full_index)
								speeds.append(mean_speed)
								if numpy.isnan(mean_speed):
									print('wtf',full_index)
	
	index_list_selectable = numpy.array(index_list)
	speeds_selectable = numpy.array(speeds)
	
	n_cells = len(index_list)
	print('ntraj=',n_cells)
	speed_bins = numpy.percentile(speeds, numpy.arange(0,101,10))
	xlocs,bins,nbins = binned_statistic(speeds,speeds,bins=speed_bins)
	
	corr_times_stat_bs = []
	
	for n in range(nbootstrap):
		cell_choices = numpy.random.randint(0,high=n_cells,size=n_cells)
		
		indexes_bs = index_list_selectable[cell_choices]
		speeds_bs = speeds_selectable[cell_choices]
		
		corr_times_bs = []
		for index in indexes_bs:
			
			corr_times_bs.append( numpy.mean(corr_times_coll[index])/60. )
		
		corr_time_stat,bins,nbins = binned_statistic(speeds_bs,corr_times_bs,bins=speed_bins)
		
		corr_times_stat_bs.append(corr_time_stat)
	
	corr_times_stat_bs = numpy.array(corr_times_stat_bs)
	
	corr_times_med = numpy.percentile(corr_times_stat_bs,50,axis=0)
	lb = numpy.percentile(corr_times_stat_bs,5,axis=0)
	ub = numpy.percentile(corr_times_stat_bs,95,axis=0)
	
	###
	P_fit, pcov = curve_fit(linmod,xlocs,corr_times_med,sigma=ub-lb)
	
	taus = [9*60,1080,1080+9*60,1080+18*60,1080+27*60]#,1080+180*30]
	
	for experiment in speeds_dict:
		if 'fish' in experiment:
			for treatment in speeds_dict[experiment]:
				if 'control' in treatment and 'highfreq' not in treatment:
					
					for tau in taus:
						msds_by_tau[tau] = []
						mean_speeds[tau] = []
						for sample in speeds_dict[experiment][treatment]:
							
							msds_by_traj = trajectory_analysis_functions_09192019.calculate_MSDs(drift_corrected_traj_dict[experiment][treatment][sample], tau, overlap=True)
							
							for traj_ind in msds_by_traj:
								
								if traj_ind in speeds_dict[experiment][treatment][sample] and len(msds_by_traj[traj_ind]) > .5 and len(speeds_dict[experiment][treatment][sample][traj_ind]) > .5:
									mean_speed=numpy.mean(speeds_dict[experiment][treatment][sample][traj_ind])
										
									msds_by_tau[tau].append( numpy.mean(msds_by_traj[traj_ind]) )
									mean_speeds[tau].append( mean_speed )
										
	n_bootstrap = 500
	
	tau_list = []
	
	xlocs_all = []
	msd_stat_all = []
	
	lb_all = []
	ub_all = []
	
	custom_cycler = cycler(color=sns.color_palette(palette='Paired'))
	
	ax.set_prop_cycle(custom_cycler)
	
	for tau in taus:
		n_cells = len(msds_by_tau[tau])
		
		msds_by_tau[tau] = numpy.array(msds_by_tau[tau])
		mean_speeds[tau] = numpy.array(mean_speeds[tau])
		print('MSD v speed, tau= ',tau,' ntraj=',len(mean_speeds[tau]))
		speed_bins = numpy.percentile(mean_speeds[tau],numpy.arange(0,101,5))
		xlocs,bins,nbins = binned_statistic(mean_speeds[tau],mean_speeds[tau],bins=speed_bins)
		
		if tau < 2000:
			xlocs0 = xlocs
		msd_stat_bs = []
		
		tau_list.append( r'$\tau=$' + str(int(tau/60)) + ' min' )
		for n in range(n_bootstrap):
			cell_choices = numpy.random.randint(0,high=n_cells,size=n_cells)
			
			msds_bs = msds_by_tau[tau][cell_choices]
			mean_speeds_bs = mean_speeds[tau][cell_choices]
			
			msd_stat,bins,nbins = binned_statistic(mean_speeds_bs,msds_bs,bins=speed_bins)
			
			msd_stat_bs.append(msd_stat)
		
		msd_stat_bs = numpy.array(msd_stat_bs)
		
		msd_stat_med = numpy.percentile(msd_stat_bs, 50, axis=0)
		lb = numpy.percentile(msd_stat_bs, 5, axis=0)
		ub = numpy.percentile(msd_stat_bs, 95, axis=0)
		
		msd_stat_med_sc = msd_stat_med/tau*60/4 ###This is the measure of the effective diffusion constant; the factor of 4 comes from the definition
		lb_sc = lb/tau*60/4
		ub_sc = ub/tau*60/4
		
		ax.set_xscale('log')
		ax.set_yscale('log')
		
		ax.errorbar(xlocs, msd_stat_med_sc, yerr = [msd_stat_med_sc - lb_sc, ub_sc - msd_stat_med_sc], linestyle='None',marker='o',markersize=2,elinewidth=.5)
		
		xlocs_all.extend(xlocs)
		msd_stat_all.extend(msd_stat_med_sc)
		lb_all.extend(lb_sc)
		ub_all.extend(ub_sc)
	
	legend1=ax.legend(tau_list,fontsize=8,loc="upper left")
	
	#lin_params,cov_mat = numpy.polyfit( numpy.log10(xlocs_all),numpy.log10(msd_stat_all),deg=1,w=1/numpy.log(numpy.array(ub_all)-numpy.array(lb_all)),full=False,cov=True )
	print(len(xlocs_all),len(msd_stat_all))
	popt_spc,pcov_spc = curve_fit( make_spc(*P_fit), xlocs_all,numpy.log10(msd_stat_all), sigma = numpy.log10(numpy.array(ub_all)/numpy.array(lb_all)),bounds=(0,numpy.inf) ) #sigma = numpy.log10(numpy.array(ub_all)-numpy.array(lb_all)),
	popt_prw,pcov_prw = curve_fit( prw, xlocs_all,numpy.log10(msd_stat_all), sigma = numpy.log10(numpy.array(ub_all)/numpy.array(lb_all)), bounds=(0,numpy.inf) ) #sigma = numpy.log10(numpy.array(ub_all)-numpy.array(lb_all)),
	
	my_spc = make_spc(*P_fit)
	
	popt_spc_free,pcov_spc_free = curve_fit( spc, xlocs_all,numpy.log10(msd_stat_all), sigma = numpy.log10(numpy.array(ub_all)/numpy.array(lb_all)), p0=[.2,.3],bounds=(0,numpy.inf) ) #sigma = numpy.log10(numpy.array(ub_all)-numpy.array(lb_all)),

	print('finished optimizing')
	print(popt_spc)
	handle2,=ax.plot(xlocs0, numpy.power(10,my_spc(xlocs0,*popt_spc)), 'k')
	#handle1,=ax.plot(xlocs0, numpy.power(10,spc(xlocs0,*popt_spc_free)), '--',color='brown',linewidth=.7)
	handle3,=ax.plot(xlocs0, numpy.power(10,prw(xlocs0,*popt_prw)), '--', color='grey',linewidth=.7)
	
	ms_5p = numpy.percentile(speeds,5)
	ms_95p = numpy.percentile(speeds,95)
	print(ms_5p,ms_95p)
	print('D range, spc, ',numpy.power(10,spc(ms_95p,*P_fit)-spc(ms_5p,*P_fit)))
	print('D range, prw, ',numpy.power(10,prw(ms_95p,*popt_prw)-prw(ms_5p,*popt_prw)))
	
	legend2 = ax.legend([handle2,handle3],['SPC','UPT'],loc="lower right")
	ax.add_artist(legend1)
	ax.set_ylabel(r'Effective diffusion constant, $D_{eff}$ ($\frac{\mu m^2}{min}$)',fontsize=9)
	ax.set_xlabel(r'Cell speed ($\frac{\mu m}{min}$)')

def MSD_v_speed_deviations(speeds_dict, corr_times_dict, drift_corrected_traj_dict, ax):
	
	def make_spc(a,b):
		def spc_fixed_ratio(x,c):
			return numpy.log10(c*x**2*(a*x + b))
		return spc_fixed_ratio
		
	def spc(x,a,b):
		return numpy.log10(a*x**2*(x + b/a))
	
	def prw(x,a):
		return numpy.log10(a*x**2)
	
	def linmod(x,a,b):
		return a*x + b
	msds_by_tau = {}
	mean_speeds = {}
	indexes = []
	
	corr_times_coll = {}
	speeds = []
	index_list = []
	
	corr_times_bootstrapped = []
	speeds_bootstrapped = []
	
	nbootstrap = 500
	
	for experiment in corr_times_dict:
		if 'fish' in experiment:
			for treatment in corr_times_dict[experiment]:
				if 'control' in treatment and 'highfreq' not in treatment:
					
					for sample in corr_times_dict[experiment][treatment]:
						
						for traj_ind in corr_times_dict[experiment][treatment][sample]:
							
							if traj_ind in speeds_dict[experiment][treatment][sample] and len(speeds_dict[experiment][treatment][sample][traj_ind]) > .5:
								
								mean_speed=numpy.mean(speeds_dict[experiment][treatment][sample][traj_ind])
									
								if ~numpy.isnan(mean_speed):
									full_index = sample + '_' + traj_ind	
									corr_times_coll[full_index] = corr_times_dict[experiment][treatment][sample][traj_ind]
									index_list.append(full_index)
									speeds.append(mean_speed)
	
	index_list_selectable = numpy.array(index_list)
	speeds_selectable = numpy.array(speeds)
	
	n_cells = len(index_list)
	speed_bins = numpy.percentile(speeds, numpy.arange(0,101,10))
	xlocs,bins,nbins = binned_statistic(speeds,speeds,bins=speed_bins)
	
	corr_times_stat_bs = []
	
	for n in range(nbootstrap):
		cell_choices = numpy.random.randint(0,high=n_cells,size=n_cells)
		
		indexes_bs = index_list_selectable[cell_choices]
		speeds_bs = speeds_selectable[cell_choices]
		
		corr_times_bs = []
		for index in indexes_bs:
			
			corr_times_bs.append( numpy.mean(corr_times_coll[index])/60. )
		
		corr_time_stat,bins,nbins = binned_statistic(speeds_bs,corr_times_bs,bins=speed_bins)
		
		corr_times_stat_bs.append(corr_time_stat)
	
	corr_times_stat_bs = numpy.array(corr_times_stat_bs)
	
	corr_times_med = numpy.percentile(corr_times_stat_bs,50,axis=0)
	lb = numpy.percentile(corr_times_stat_bs,5,axis=0)
	ub = numpy.percentile(corr_times_stat_bs,95,axis=0)
	
	###
	P_fit, pcov = curve_fit(linmod,xlocs,corr_times_med,sigma=ub-lb)
	
	msds_by_tau = {}
	mean_speeds = {}
	indexes = []
	
	taus = [9*60,1080,1080+9*60,1080+18*60,1080+27*60]#,1080+180*30]
	
	for experiment in speeds_dict:
		if 'fish' in experiment:
			for treatment in speeds_dict[experiment]:
				if 'control' in treatment and 'highfreq' not in treatment:
					
					for tau in taus:
						msds_by_tau[tau] = []
						mean_speeds[tau] = []
						for sample in speeds_dict[experiment][treatment]:
							
							msds_by_traj = trajectory_analysis_functions_09192019.calculate_MSDs(drift_corrected_traj_dict[experiment][treatment][sample], tau, overlap=True)
							
							for traj_ind in msds_by_traj:
								
									
								if traj_ind in speeds_dict[experiment][treatment][sample] and len(msds_by_traj[traj_ind]) > .5 and len(speeds_dict[experiment][treatment][sample][traj_ind]) > .5:
									mean_speed=numpy.mean(speeds_dict[experiment][treatment][sample][traj_ind])
										
									msds_by_tau[tau].append( numpy.mean(msds_by_traj[traj_ind]) )
									mean_speeds[tau].append( mean_speed )
										
	n_bootstrap = 500
	
	tau_list = []
	
	xlocs_all = []
	msd_stat_all = []
	
	lb_all = []
	ub_all = []
	
	custom_cycler = cycler(color=sns.color_palette(palette='Paired'))
	
	ax.set_prop_cycle(custom_cycler)
	
	xlocs_by_tau = []
	msd_med_by_tau = []
	msd_lbs_by_tau = []
	msd_ubs_by_tau = []
	
	for tau in taus:
		n_cells = len(msds_by_tau[tau])
		
		msds_by_tau[tau] = numpy.array(msds_by_tau[tau])
		mean_speeds[tau] = numpy.array(mean_speeds[tau])
		
		speed_bins = numpy.percentile(mean_speeds[tau],numpy.arange(0,101,5))
		xlocs,bins,nbins = binned_statistic(mean_speeds[tau],mean_speeds[tau],bins=speed_bins)
		
		if tau < 2000:
			xlocs0 = xlocs
		msd_stat_bs = []
		
		tau_list.append( r'$\tau=$' + str(int(tau/60)) + ' min' )
		for n in range(n_bootstrap):
			cell_choices = numpy.random.randint(0,high=n_cells,size=n_cells)
			
			msds_bs = msds_by_tau[tau][cell_choices]
			mean_speeds_bs = mean_speeds[tau][cell_choices]
			
			msd_stat,bins,nbins = binned_statistic(mean_speeds_bs,msds_bs,bins=speed_bins)
			
			msd_stat_bs.append(msd_stat)
		
		msd_stat_bs = numpy.array(msd_stat_bs)
		
		msd_stat_med = numpy.percentile(msd_stat_bs, 50, axis=0)
		lb = numpy.percentile(msd_stat_bs, 5, axis=0)
		ub = numpy.percentile(msd_stat_bs, 95, axis=0)
		
		
		
		#ax.errorbar(xlocs, msd_stat_med/tau*60, yerr = [(msd_stat_med - lb)/tau*60, (ub - msd_stat_med)/tau*60], linestyle='None',marker='o',markersize=3,elinewidth=.5)
		
		xlocs_all.extend(xlocs)
		msd_stat_all.extend(msd_stat_med/tau*60)
		lb_all.extend(lb/tau*60)
		ub_all.extend(ub/tau*60)
		
		xlocs_by_tau.append(xlocs)
		msd_med_by_tau.append(msd_stat_med/tau*60)
		msd_lbs_by_tau.append(lb/tau*60)
		msd_ubs_by_tau.append(ub/tau*60)
	
	ax.set_xscale('log')
	ax.set_yscale('log')
		
	xlocs_by_tau = numpy.array(xlocs_by_tau)
	msd_med_by_tau = numpy.array(msd_med_by_tau)
	msd_lbs_by_tau = numpy.array(msd_lbs_by_tau)
	msd_ubs_by_tau = numpy.array(msd_ubs_by_tau)
	
	#legend1=ax.legend(tau_list,fontsize=8)
	
	lin_params,cov_mat = numpy.polyfit( numpy.log10(xlocs_all),numpy.log10(msd_stat_all),deg=1,w=1/numpy.log(numpy.array(ub_all)-numpy.array(lb_all)),full=False,cov=True )
	
	popt_spc,pcov_spc = curve_fit( make_spc(*P_fit), xlocs_all,numpy.log10(msd_stat_all), sigma = numpy.log10(numpy.array(ub_all)/numpy.array(lb_all)), bounds=(0,numpy.inf) ) #sigma = numpy.log10(numpy.array(ub_all)-numpy.array(lb_all)),
	popt_prw,pcov_prw = curve_fit( prw, xlocs_all,numpy.log10(msd_stat_all), sigma = numpy.log10(numpy.array(ub_all)/numpy.array(lb_all)), bounds=(0,numpy.inf) ) #sigma = numpy.log10(numpy.array(ub_all)-numpy.array(lb_all)),
	
	my_spc = make_spc(*P_fit)
	
	quad_pred = prw(xlocs_by_tau,*popt_prw)
	cub_pred = my_spc(xlocs_by_tau,*popt_spc)
	cub_nfp_pred = spc(xlocs_by_tau,*P_fit)
	
	deviations = msd_med_by_tau/numpy.power(10,quad_pred)
	model_deviations = numpy.power(10,cub_pred-quad_pred)
	model2_deviations = numpy.power(10,cub_nfp_pred-quad_pred)
	
	lb_deviations = msd_lbs_by_tau/numpy.power(10,quad_pred)
	ub_deviations = msd_ubs_by_tau/numpy.power(10,quad_pred)
	
	for i in range(len(taus)):
		ax.errorbar(xlocs_by_tau[i,:],deviations[i,:], yerr = [deviations[i,:] - lb_deviations[i,:], ub_deviations[i,:] - deviations[i,:]], linestyle='None',marker='o',markersize=2,elinewidth=.5)
	
	
	
	
	#handle2,=ax.plot(xlocs_by_tau[0,:], model_deviations[0,:], '--',color='brown',linewidth=.7)
	handle1,=ax.plot(xlocs_by_tau[0,:], model_deviations[0,:], 'k')
	handle3,=ax.plot(xlocs0, numpy.ones(len(xlocs0)), '--',color='grey',linewidth=.7)
	
	
	#legend2 = pt.legend([handle1,handle2],['SPC','UPT'],loc=4)
	#ax.add_artist(legend1)
	ax.set_ylabel(r'$\frac{D_{eff}}{D_{UPT}}$')
	ax.set_xticklabels([])
	#ax.set_xlabel('Cell speed ($\mu$m/min)')
	
def MSD_overall(speeds_dict, drift_corrected_traj_dict, ax, inset_ax):
	
	msds_by_tau = {}
	mean_speeds = {}
	indexes_by_tau = {}
	
	taus = [45,90,135,180,225,270,315,360,450,540,630,720,810,900,1080,1080+180*1,1080+180*2,1080+180*3,1080+180*4,1080+180*6,1080+180*8,1080+180*10,1080+180*12,1080+180*14,1080+180*18,1080+180*22]#numpy.arange(0,45*100,45)

	for experiment in speeds_dict:
		if 'fish' in experiment:
			for treatment in speeds_dict[experiment]:
				if 'control' in treatment and 'highfreq' not in treatment:
					
					for tau in taus:
						msds_by_tau[tau] = {}
						mean_speeds[tau] = {}
						indexes_by_tau[tau] = []
						for sample in speeds_dict[experiment][treatment]:
							#print('about to calculate msds')
							msds_by_traj = trajectory_analysis_functions_09192019.calculate_MSDs(drift_corrected_traj_dict[experiment][treatment][sample], tau, overlap=True)
							#print('done calculating msds')
							for traj_ind in msds_by_traj:
								
								if traj_ind in speeds_dict[experiment][treatment][sample] and len(msds_by_traj[traj_ind]) > .5 and len(speeds_dict[experiment][treatment][sample][traj_ind]) > .5:
									mean_speed=numpy.mean(speeds_dict[experiment][treatment][sample][traj_ind])
									
									full_index = sample + '_' + traj_ind
									
									msds_by_tau[tau][full_index] = numpy.mean(msds_by_traj[traj_ind])
									mean_speeds[tau][full_index] = mean_speed
									indexes_by_tau[tau].append(full_index)
							#print('done averaging msds')
	inds0 = set(indexes_by_tau[taus[0]])
	
	for tau in taus[1:]:
		inds0.intersection_update(set(indexes_by_tau[tau]))
	
	msds_by_tau_arr = []
	
	n_cells = len(inds0)
	print('MSD overall calculation, ', n_cells)
	
	for tau in taus:
		msds_temp = []
		for index in inds0:
			msds_temp.append( msds_by_tau[tau][index] )
		msds_by_tau_arr.append(msds_temp)
	
	msds_by_tau_arr = numpy.array(msds_by_tau_arr)
	
	print(msds_by_tau_arr.shape)
	
	n_bootstrap = 500
	
	mean_msds_bs_list = []
	for n in range(n_bootstrap):
	
		cell_choices = numpy.random.randint(0,high=n_cells,size=n_cells)
		
		mean_msds_bs = numpy.mean(msds_by_tau_arr[:,cell_choices],axis=1)
		
		mean_msds_bs_list.append(mean_msds_bs)
	
	med_stat = numpy.percentile(mean_msds_bs_list,50,axis=0)
	lb = numpy.percentile(mean_msds_bs_list,5,axis=0)
	ub = numpy.percentile(mean_msds_bs_list,95,axis=0)
	
	ax.set_xscale('log')
	ax.set_yscale('log')
	
	taus = numpy.array(taus)/60 #times in min
	
	early_t = taus[taus<10]
	early_msd = med_stat[taus<10]
	lb_early = lb[taus<10]
	ub_early = ub[taus<10]
	
	late_t = taus[taus >= 15]
	late_msd = med_stat[taus >= 15]
	lb_late = lb[taus>=15]
	ub_late = ub[taus>=15]
	
	wlc_fit = curve_fit(WLC_MSD,taus,med_stat,p0=[2,1,1],sigma=ub-lb)[0]
	
	params_early = numpy.polyfit(numpy.log10(early_t),numpy.log10(early_msd),deg=1,w=1/numpy.log10(ub_early/lb_early))
	params_late = numpy.polyfit(numpy.log10(late_t),numpy.log10(late_msd),deg=1,w=1/numpy.log10(ub_late/lb_late))
	
	ax.errorbar(taus,med_stat,yerr=[med_stat-lb,ub-med_stat],marker='o',linestyle='None',zorder=1,markersize=5,alpha=.8)
	
	print(params_early)
	print(params_late)
	
	print(wlc_fit)
	
	handle=ax.plot(taus, WLC_MSD(taus,*wlc_fit),'k',zorder=2)
	
	
	ax.plot(early_t, 1.5*numpy.power(10,params_early[0]*numpy.log10(early_t) + params_early[1]), 'k--',zorder=2)
	yvals = 1.5*numpy.power(10,params_early[0]*numpy.log10(early_t) + params_early[1])
	
	ax.plot([early_t[2],early_t[2]],[yvals[2],yvals[3]],'k')
	ax.plot([early_t[2],early_t[3]],[yvals[3],yvals[3]],'k')
	
	#ax.plot(late_t, numpy.power(10,numpy.log10(late_t) + params_late[1]), 'k--',zorder=2)
	#yvals = 1.5*numpy.power(10,numpy.log10(late_t) + params_late[1])
	
	#ax.plot([late_t[2],late_t[2]],[yvals[2],yvals[3]],'k')
	#ax.plot([late_t[2],late_t[3]],[yvals[3],yvals[3]],'k')
	#ax.text(40,7000,str(round(params_late[0],1)),horizontalalignment='center')
	ax.text(1.6,300,str(round(params_early[0],1)),horizontalalignment='center')
	
	
	#ax.plot(late_t, numpy.power(10,params_late[0]*numpy.log10(late_t) + params_late[1]), 'k',zorder=2)
	
	ax.set_ylabel(r'$\langle MSD(\tau) \rangle$ ($\mu m^2$)')
	ax.set_xlabel(r'$\tau$ (min)')
	
	inset_ax.errorbar(early_t,early_msd,yerr=[early_msd-lb_early,ub_early-early_msd],marker='o',markersize=2,elinewidth=.3,linewidth=.5)
	inset_ax.tick_params(labelsize=6,length=1)
	inset_ax.set_ylabel(r'$\langle MSD(\tau) \rangle$',fontsize=8)
	inset_ax.set_xlabel(r'$\tau$ (min)',fontsize=8)
	ax.legend(handle,['PRW Fit'])
	#ax.legend(handle,['GPRW Fit'])

def MSD_overall_shorter(speeds_dict, drift_corrected_traj_dict, ax, inset_ax):
	
	msds_by_tau = {}
	mean_speeds = {}
	indexes_by_tau = {}
	
	taus = [45,90,135,180,225,270,315,360,450,540,630,720,810,900]#,1080+180*2,1080+180*3,1080+180*4,1080+180*6,1080+180*8,1080+180*10,1080+180*12,1080+180*14,1080+180*18,1080+180*22]#numpy.arange(0,45*100,45)

	for experiment in speeds_dict:
		if 'fish' in experiment:
			for treatment in speeds_dict[experiment]:
				if 'control' in treatment and 'highfreq' not in treatment:
					
					for tau in taus:
						msds_by_tau[tau] = {}
						mean_speeds[tau] = {}
						indexes_by_tau[tau] = []
						for sample in speeds_dict[experiment][treatment]:
							#print('about to calculate msds')
							msds_by_traj = trajectory_analysis_functions_09192019.calculate_MSDs(drift_corrected_traj_dict[experiment][treatment][sample], tau, overlap=True)
							#print('done calculating msds')
							for traj_ind in msds_by_traj:
								
								if traj_ind in speeds_dict[experiment][treatment][sample] and len(msds_by_traj[traj_ind]) > .5 and len(speeds_dict[experiment][treatment][sample][traj_ind]) > .5:
									mean_speed=numpy.mean(speeds_dict[experiment][treatment][sample][traj_ind])
									
									full_index = sample + '_' + traj_ind
									
									msds_by_tau[tau][full_index] = numpy.mean(msds_by_traj[traj_ind])
									mean_speeds[tau][full_index] = mean_speed
									indexes_by_tau[tau].append(full_index)
							#print('done averaging msds')
	inds0 = set(indexes_by_tau[taus[0]])
	
	for tau in taus[1:]:
		inds0.intersection_update(set(indexes_by_tau[tau]))
	
	msds_by_tau_arr = []
	
	n_cells = len(inds0)
	print('MSD overall calculation, shorter time window ', n_cells)
	
	for tau in taus:
		msds_temp = []
		for index in inds0:
			msds_temp.append( msds_by_tau[tau][index] )
		msds_by_tau_arr.append(msds_temp)
	
	msds_by_tau_arr = numpy.array(msds_by_tau_arr)
	
	print(msds_by_tau_arr.shape)
	
	n_bootstrap = 500
	
	mean_msds_bs_list = []
	for n in range(n_bootstrap):
	
		cell_choices = numpy.random.randint(0,high=n_cells,size=n_cells)
		
		mean_msds_bs = numpy.mean(msds_by_tau_arr[:,cell_choices],axis=1)
		
		mean_msds_bs_list.append(mean_msds_bs)
	
	med_stat = numpy.percentile(mean_msds_bs_list,50,axis=0)
	lb = numpy.percentile(mean_msds_bs_list,5,axis=0)
	ub = numpy.percentile(mean_msds_bs_list,95,axis=0)
	
	ax.set_xscale('log')
	ax.set_yscale('log')
	
	taus = numpy.array(taus)/60 #times in min
	
	early_t = taus[taus<10]
	early_msd = med_stat[taus<10]
	lb_early = lb[taus<10]
	ub_early = ub[taus<10]
	
	late_t = taus[taus >= 15]
	late_msd = med_stat[taus >= 15]
	lb_late = lb[taus>=15]
	ub_late = ub[taus>=15]
	
	wlc_fit = curve_fit(WLC_MSD,taus,med_stat,p0=[2,1,1],sigma=ub-lb)[0]
	
	params_early = numpy.polyfit(numpy.log10(early_t),numpy.log10(early_msd),deg=1,w=1/numpy.log10(ub_early/lb_early))
	params_late = numpy.polyfit(numpy.log10(late_t),numpy.log10(late_msd),deg=1,w=1/numpy.log10(ub_late/lb_late))
	
	ax.errorbar(taus,med_stat,yerr=[med_stat-lb,ub-med_stat],marker='o',linestyle='None',zorder=1,markersize=5,alpha=.8)
	
	print(params_early)
	print(params_late)
	
	print(wlc_fit)
	
	handle=ax.plot(taus, WLC_MSD(taus,*wlc_fit),'k',zorder=2)
	
	
	ax.plot(early_t, 1.5*numpy.power(10,params_early[0]*numpy.log10(early_t) + params_early[1]), 'k--',zorder=2)
	yvals = 1.5*numpy.power(10,params_early[0]*numpy.log10(early_t) + params_early[1])
	
	ax.plot([early_t[2],early_t[2]],[yvals[2],yvals[3]],'k')
	ax.plot([early_t[2],early_t[3]],[yvals[3],yvals[3]],'k')
	
	#ax.plot(late_t, numpy.power(10,numpy.log10(late_t) + params_late[1]), 'k--',zorder=2)
	#yvals = 1.5*numpy.power(10,numpy.log10(late_t) + params_late[1])
	
	#ax.plot([late_t[2],late_t[2]],[yvals[2],yvals[3]],'k')
	#ax.plot([late_t[2],late_t[3]],[yvals[3],yvals[3]],'k')
	#ax.text(40,7000,str(round(params_late[0],1)),horizontalalignment='center')
	ax.text(1.6,300,str(round(params_early[0],1)),horizontalalignment='center')
	
	
	#ax.plot(late_t, numpy.power(10,params_late[0]*numpy.log10(late_t) + params_late[1]), 'k',zorder=2)
	
	ax.set_ylabel(r'$\langle MSD(\tau) \rangle$ ($\mu m^2$)')
	ax.set_xlabel(r'$\tau$ (min)')
	
	inset_ax.errorbar(early_t,early_msd,yerr=[early_msd-lb_early,ub_early-early_msd],marker='o',markersize=2,elinewidth=.3,linewidth=.5)
	inset_ax.tick_params(labelsize=6,length=1)
	inset_ax.set_ylabel(r'$\langle MSD(\tau) \rangle$',fontsize=8)
	inset_ax.set_xlabel(r'$\tau$ (min)',fontsize=8)
	ax.legend(handle,['PRW Fit'])
	#ax.legend(handle,['GPRW Fit'])



def speed_angle_heterogeneity_panels_revised(speeds_dict, trajectory_dict_polar, speed_angle_coupling_dict, axis_lists):
	
	####Collect overall distributions of speeds and angles
	
	all_speeds = []
	all_angles = []
	
	focal_sample = 'lckgfp_dob07072019_fish3_488_trial3_3' 
	individual_trajectories = {}
	#chosen_indexes = {'15','24','11','30'}
	chosen_indexes = ['24','30','11','15']
	#chosen_indexes = ['5','20','11','15']
	#chosen_indexes = {15,20,11,5}
	color_cycle = {'24':'goldenrod','15':'C6','30':'C9','11':'C5'}
	
	speed_classes = numpy.array([0,3,6,11,18])
	for experiment in trajectory_dict_polar:
		if 'fish' in experiment:
			for treatment in trajectory_dict_polar[experiment]:
				if 'control' in treatment:
				#if 'control' in treatment and 'highfreq' not in treatment:# and 'highfreq' not in treatment:
					for sample in trajectory_dict_polar[experiment][treatment]:
						for traj_ind in trajectory_dict_polar[experiment][treatment][sample]:
							
							if traj_ind in speed_angle_coupling_dict[experiment][treatment][sample]:
								traj_data = trajectory_dict_polar[experiment][treatment][sample][traj_ind]
								relative_angle_data = speed_angle_coupling_dict[experiment][treatment][sample][traj_ind]
								relative_angles = numpy.array([entry[1] for entry in relative_angle_data])
								
								all_speeds.extend( numpy.array(traj_data[1])/(traj_data[0][1]-traj_data[0][0])*60 )
								all_angles.extend( relative_angles )
								
								mean_speed = numpy.mean( numpy.array(traj_data[1])/(traj_data[0][1]-traj_data[0][0])*60 )
								#if sample == focal_sample and traj_ind in chosen_indexes:
								#if len(individual_trajectories.keys()) < 4 and ~numpy.isnan(mean_speed):
									
									#speed_class = numpy.argmax( mean_speed - speed_classes < 0 ) - 1
									#if speed_class not in individual_trajectories and (traj_data[0][-1] - traj_data[0][0])/45*.75 > 130 and len(traj_data[0]) - (traj_data[0][-1] - traj_data[0][0])/45 > -.5:
										
								if sample == focal_sample and traj_ind in chosen_indexes:		
									individual_trajectories[traj_ind] = {}
									individual_trajectories[traj_ind]['speeds'] = numpy.array(traj_data[1][:226])/48*60
									individual_trajectories[traj_ind]['angles'] = relative_angles
									individual_trajectories[traj_ind]['times'] = numpy.array(traj_data[0][:226]/60)
									individual_trajectories[traj_ind]['vx'] = individual_trajectories[traj_ind]['speeds']*numpy.cos(numpy.array(traj_data[2][:226]))
							elif sample == focal_sample and traj_ind in chosen_indexes:
								print(traj_ind,'why')
	#print(individual_trajectories.keys())
	angle_bins = numpy.arange(0,numpy.pi + .01,numpy.pi/10)
	speed_bins = numpy.arange(0,30.5,2)
	panel_ind = 0
	print('Speed dist, number of speed measurements, ', len(all_speeds))
	print('Angle dist, number of angle measurements, ', len(all_angles))
	for traj_ind in chosen_indexes:
		
		ax_subset = axis_lists[panel_ind]
		traj_len = len(individual_trajectories[traj_ind]['vx'])
		
		ax_subset[0].plot( individual_trajectories[traj_ind]['times'], individual_trajectories[traj_ind]['vx'],color=color_cycle[traj_ind] )
		#ax_subset[0].set_xlabel('Time (min)')
		ax_subset[0].set_ylabel(r'$v_x$ ($\frac{\mu m}{min}$)')
		
		ax_subset[0].set_ylim(-30,30)
		
		sns.distplot(individual_trajectories[traj_ind]['speeds'], bins=speed_bins, norm_hist=True,ax=ax_subset[1],color=color_cycle[traj_ind])
		sns.distplot(all_speeds, bins=speed_bins, norm_hist=True,color='grey',ax=ax_subset[1],kde_kws={"zorder":0},hist_kws={"alpha": .7, "zorder": 0})
		ax_subset[1].set_xlim([0,30])
		
		sns.distplot(individual_trajectories[traj_ind]['angles'], bins=angle_bins, norm_hist=True,ax=ax_subset[2],color=color_cycle[traj_ind])
		sns.distplot(all_angles, bins=angle_bins, norm_hist=True,color='grey',ax=ax_subset[2],kde_kws={"zorder":0},hist_kws={"alpha": .7, "zorder": 0})
		ax_subset[2].set_xlim([0,numpy.pi])
		
		if panel_ind == 3:
			ax_subset[0].set_xlabel('Time (min)')
			ax_subset[1].set_xlabel(r'Speed ($\frac{\mu m}{min}$)')
			ax_subset[2].set_xlabel('Turn angle (rad)')
		else:
			ax_subset[0].set_xticklabels([])
			ax_subset[1].set_xticklabels([])
			ax_subset[2].set_xticklabels([])
		panel_ind +=1	
		

def angle_speed_correlations_with_perturbations(speeds_dict,relative_angle_dict,experiment,ax,ax2):
	
	treatment_list = []
	treatment_label_list = []
	experiment_titles = {'fish_T':'D. rerio','Gerard_mouse_T':'M. musculus','Gautreau_dicty':'Dictyostelium'}

	study_labels = {'fish_T':'This study','Gerard_mouse':'Gerard et al.','Gautreau_dicty':'Dang et al.'}
	treatment_label_dict = {'control':'Control','rockout':'12 $\mu$M Rockout','Myo1g_KO':'Myo1g KO','Arp_KO':'Arpin KO','Arp_rescue':'Arpin rescue','WT':'WT'}
	
	angles_subset = {}
	speeds_subset = {}
	
	for treatment in relative_angle_dict[experiment]:
		if 'highfreq' not in treatment:
		
			angles_subset[treatment] = []
			speeds_subset[treatment] = []
			for sample in  relative_angle_dict[experiment][treatment]:
				for traj_ind in  relative_angle_dict[experiment][treatment][sample]:
					if traj_ind in speeds_dict[experiment][treatment][sample]:
						mean_speed=numpy.mean(speeds_dict[experiment][treatment][sample][traj_ind])
						good_steps = len(speeds_dict[experiment][treatment][sample][traj_ind])
						if 'fish' in experiment:
							if good_steps >= 30:
								angles_subset[treatment].append( numpy.mean(numpy.cos([entry[1] for entry in relative_angle_dict[experiment][treatment][sample][traj_ind]]) ))
								speeds_subset[treatment].append( mean_speed )
						elif ~numpy.isnan(mean_speed) and good_steps > 10:
							
							angles_subset[treatment].append( numpy.mean(numpy.cos([entry[1] for entry in relative_angle_dict[experiment][treatment][sample][traj_ind]]) ))
							speeds_subset[treatment].append( mean_speed )
						
	corr_times_stat_bs = {}
	
	n_bootstrap = 500
	
	for treatment in angles_subset:
		if 'highfreq' not in treatment:
			treatment_list.append(treatment)
			treatment_label_list.append(treatment_label_dict[treatment])
			if 'fish' in experiment and 'control' in treatment:
				speed_bins = numpy.percentile(speeds_subset[treatment],numpy.arange(0,101,10))
			else:
				speed_bins = numpy.percentile(speeds_subset[treatment],numpy.arange(0,101,20))
			angles_subset[treatment] = numpy.array(angles_subset[treatment])
			speeds_subset[treatment] = numpy.array(speeds_subset[treatment])
			xlocs,bins,nbins = binned_statistic(speeds_subset[treatment],speeds_subset[treatment],bins=speed_bins)
			n_cells = len(angles_subset[treatment])
			print('Speed angle corr, ', treatment, ' ntraj=',n_cells)
			ct_stat_bs = []
			for n in range(n_bootstrap):
				cell_choices = numpy.random.randint(0,high=n_cells,size=n_cells)
				corr_times_temp = angles_subset[treatment][cell_choices]
				speeds_temp = speeds_subset[treatment][cell_choices]
				
				ct_stat,bins,nbins = binned_statistic(speeds_temp,corr_times_temp,bins=speed_bins)
				
				ct_stat_bs.append(ct_stat)
		
		ct_stat_bs = numpy.array(ct_stat_bs)
		
		ct_stat_med = numpy.percentile(ct_stat_bs,50,axis=0)
		lb = numpy.percentile(ct_stat_bs,5,axis=0)
		ub = numpy.percentile(ct_stat_bs,95,axis=0)
		
		ax.errorbar(xlocs,ct_stat_med,yerr=[ct_stat_med-lb,ub-ct_stat_med],marker='o')
	ax.legend(treatment_label_list)
		
	ax.set_xlabel('Speed ($\mu$m/min)')
	ax.set_ylabel(r'$\langle Cos\theta \rangle$')
	
	for treatment in treatment_list:
		sns.distplot(speeds_subset[treatment],ax=ax2)
	
	ax2.set_xlabel('Speed ($\mu$m/min)')
	
def angle_speed_correlations_with_perturbations_all_cells(speeds_dict,relative_angle_dict,experiment,ax,ax2):
	import scipy.signal,scipy.ndimage
	def percentile75(x):
		return numpy.percentile(x,75)
	def percentile25(x):
		return numpy.percentile(x,25)
	treatment_list = []
	treatment_label_list = []
	experiment_titles = {'fish_T':'D. rerio','Gerard_mouse_T':'M. musculus','Gautreau_dicty':'Dictyostelium'}

	study_labels = {'fish_T':'This study','Gerard_mouse_T':'Gerard et al.','Gautreau_dicty':'Dang et al.'}
	treatment_label_dict = {'control':'Control','rockout':'12 $\mu$M Rockout','Myo1g_KO':'Myo1g KO','Arp_KO':'Arpin KO','Arp_rescue':'Arpin rescue','WT':'WT'}
	treatment_color_dict = {'control':'C0','rockout':'C2','Myo1g_KO':'C2','Arp_KO':'C2','Arp_rescue':'C3','WT':'C0'}
	treatment_color_dict_outlines = {'control':'darkblue','rockout':'darkgreen','Myo1g_KO':'darkgreen','Arp_KO':'darkgreen','Arp_rescue':'Brown','WT':'darkblue'}
	angles_subset = {}
	speeds_subset = {}
	
	angles_se = {}
	speeds_se = {}
	
	n_bootstrap = 500
	
	for treatment in relative_angle_dict[experiment]:
		if 'highfreq' not in treatment:
		
			angles_subset[treatment] = []
			speeds_subset[treatment] = []
			angles_se[treatment] = []
			speeds_se[treatment] = []
			
			for sample in  relative_angle_dict[experiment][treatment]:
				if experiment == 'Gerard_mouse_T' and treatment=='control':
					print(len(speeds_dict[experiment][treatment][sample]))
				for traj_ind in  speeds_dict[experiment][treatment][sample]:
					#if experiment == 'Gerard_mouse_T' and treatment=='control':
					
						#if traj_ind in speeds_dict[experiment][treatment][sample] and traj_ind not in relative_angle_dict[experiment][treatment][sample]:
							#print(traj_ind,speeds_dict[experiment][treatment][sample][traj_ind])
					if traj_ind in relative_angle_dict[experiment][treatment][sample]:
						mean_speed=numpy.mean(speeds_dict[experiment][treatment][sample][traj_ind])
						speed_se = numpy.std(speeds_dict[experiment][treatment][sample][traj_ind])/numpy.sqrt(len(speeds_dict[experiment][treatment][sample][traj_ind]) - 1)
						good_steps = len(speeds_dict[experiment][treatment][sample][traj_ind])
						
						if ~numpy.isnan(mean_speed):# and good_steps > 8:
							cosangles = numpy.cos([entry[1] for entry in relative_angle_dict[experiment][treatment][sample][traj_ind]])
							angles_subset[treatment].append( numpy.mean(cosangles))
							angles_se_temp = numpy.std(cosangles)/numpy.sqrt(len(cosangles) - 1)
							speeds_subset[treatment].append( mean_speed )
							
							speeds_se[treatment].append( speed_se )
							angles_se[treatment].append( angles_se_temp )
		#elif 'highfreq' in treatment: ###Append these to control samples
			#for sample in  relative_angle_dict[experiment][treatment]:
				#if experiment == 'Gerard_mouse_T' and treatment=='control':
					#print(len(speeds_dict[experiment][treatment][sample]))
				#for traj_ind in  speeds_dict[experiment][treatment][sample]:
					#if experiment == 'Gerard_mouse_T' and treatment=='control':
					
						#if traj_ind in speeds_dict[experiment][treatment][sample] and traj_ind not in relative_angle_dict[experiment][treatment][sample]:
							#print(traj_ind,speeds_dict[experiment][treatment][sample][traj_ind])
					#if traj_ind in relative_angle_dict[experiment][treatment][sample]:
						#mean_speed=numpy.mean(speeds_dict[experiment][treatment][sample][traj_ind])
						#speed_se = numpy.std(speeds_dict[experiment][treatment][sample][traj_ind])/numpy.sqrt(len(speeds_dict[experiment][treatment][sample][traj_ind]) - 1)
						#good_steps = len(speeds_dict[experiment][treatment][sample][traj_ind])
						
						#if ~numpy.isnan(mean_speed):# and good_steps > 8:
							#cosangles = numpy.cos([entry[1] for entry in relative_angle_dict[experiment][treatment][sample][traj_ind]])
							#angles_subset['control'].append( numpy.mean(cosangles))
							#angles_se_temp = numpy.std(cosangles)/numpy.sqrt(len(cosangles) - 1)
							#speeds_subset['control'].append( mean_speed )
							
							#speeds_se['control'].append( speed_se )
							#angles_se['control'].append( angles_se_temp )
			
	
	for treatment in angles_subset:
		if 'highfreq' not in treatment:
			treatment_list.append(treatment)
			treatment_label_list.append(treatment_label_dict[treatment])
			speed_bins = numpy.arange(0,16.1,2)
			#if 'fish' in experiment and 'control' in treatment:
			#	speed_bins = numpy.arange(0,18.1,2)#numpy.percentile(speeds_subset[treatment],numpy.arange(0,101,10))
			#else:
			#	speed_bins = numpy.percentile(speeds_subset[treatment],numpy.arange(0,101,20))
			angles_subset[treatment] = numpy.array(angles_subset[treatment])
			speeds_subset[treatment] = numpy.array(speeds_subset[treatment])
			print(treatment,' ntraj=',len(speeds_subset[treatment]))
			speed_order = numpy.argsort(speeds_subset[treatment])
			angles_ordered = angles_subset[treatment][speed_order]
			speeds_ordered = speeds_subset[treatment][speed_order]
			rolling_median_angles = scipy.ndimage.gaussian_filter(angles_ordered,sigma=10)#scipy.signal.medfilt(angles_ordered,kernel_size=51)
			rolling_median_speeds = scipy.ndimage.gaussian_filter(speeds_ordered,sigma=10)
			
			#ax.plot(rolling_median_speeds,rolling_median_angles,color=treatment_color_dict[treatment],zorder=2)
			
			xlocs,bins,nbins = binned_statistic(speeds_subset[treatment],speeds_subset[treatment],bins=speed_bins)
			mean_ang,bins,nbins = binned_statistic(speeds_subset[treatment],angles_subset[treatment],bins=speed_bins,statistic='median')
			perc75,bins,nbins = binned_statistic(speeds_subset[treatment],angles_subset[treatment],bins=speed_bins,statistic=percentile75)
			perc25,bins,nbins = binned_statistic(speeds_subset[treatment],angles_subset[treatment],bins=speed_bins,statistic=percentile25)
			
			n_cells = len(angles_subset[treatment])
			print(xlocs)
			print(speed_bins)
			print(mean_ang)
			#ax.plot(xlocs[:-1],mean_ang[:-1],color=treatment_color_dict[treatment],zorder=2,linewidth=2)
			#ax.plot(xlocs,perc90,color=treatment_color_dict[treatment],zorder=2,linewidth=.5)
			#ax.plot(xlocs,perc10,color=treatment_color_dict[treatment],zorder=2,linewidth=.5)
			ct_stat_bs = []
			for n in range(n_bootstrap):
				cell_choices = numpy.random.randint(0,high=n_cells,size=n_cells)
				corr_times_temp = angles_subset[treatment][cell_choices]
				speeds_temp = speeds_subset[treatment][cell_choices]
				
				ct_stat,bins,nbins = binned_statistic(speeds_temp,corr_times_temp,bins=speed_bins,statistic='median')
				
				ct_stat_bs.append(ct_stat)
		
		ct_stat_bs = numpy.array(ct_stat_bs)
		
		#ct_stat_med = numpy.percentile(ct_stat_bs,50,axis=0)
		lb = numpy.percentile(ct_stat_bs,5,axis=0)
		ub = numpy.percentile(ct_stat_bs,95,axis=0)
		
		if experiment != 'Gautreau_dicty':
			ax.errorbar(xlocs,mean_ang,yerr=[mean_ang-lb,ub-mean_ang],marker='o',color=treatment_color_dict_outlines[treatment],zorder=2,markeredgecolor=treatment_color_dict_outlines[treatment])
		else:
			xlocs = xlocs[~numpy.isnan(mean_ang)][:3]
			
			lb = lb[~numpy.isnan(mean_ang)][:3]
			ub = ub[~numpy.isnan(mean_ang)][:3]
			mean_ang = mean_ang[~numpy.isnan(mean_ang)][:3]
			ax.errorbar(xlocs,mean_ang,yerr=[mean_ang-lb,ub-mean_ang],marker='o',color=treatment_color_dict_outlines[treatment],zorder=2,markeredgecolor=treatment_color_dict_outlines[treatment])
	for treatment in angles_subset:
		if 'highfreq' not in treatment:
			treatment_list.append(treatment)
			#treatment_label_list.append(treatment_label_dict[treatment])
			#ax.errorbar(speeds_subset[treatment],angles_subset[treatment],xerr=speeds_se[treatment],yerr=angles_se[treatment],marker='o',markersize=2,alpha=.4,elinewidth=.3,linestyle='None')
			ax.errorbar(speeds_subset[treatment],angles_subset[treatment],marker='o',markersize=3,alpha=.7,elinewidth=.3,linestyle='None',color=treatment_color_dict[treatment],zorder=0)

	ax.legend(treatment_label_list)
		
	#ax3.set_xlabel(r'$\langle s \rangle_{cell} (\mu$m/min)')
	ax.set_ylabel(r'$\langle Cos\theta \rangle_{cell}$')
	ax2.set_ylabel(r'$\langle Cos\theta \rangle_{cell}$')
	for treatment in treatment_list:
		#ax2.hist(speeds_subset[treatment],histtype="step",color=treatment_color_dict[treatment],density=True,linewidth=2)
		sns.distplot(speeds_subset[treatment],ax=ax2,kde=False,norm_hist=True,color=treatment_color_dict[treatment],hist_kws={"histtype": "step", "linewidth": 2,"alpha": 1, "color": treatment_color_dict[treatment]})
	ax.set_xlabel('Cell speed ($\mu$m/min)')
	ax2.set_xlabel('Cell speed ($\mu$m/min)')
	ax2.set_ylabel('Probability density')
	if experiment=='fish_T' or experiment=='Gerard_mouse_T':
		ax.set_xlim(0,18.5)
		ax2.set_xlim(0,18.5)
	else:
		ax.set_xlim(0,9)
		ax2.set_xlim(0,9)

def angle_speed_correlations_with_perturbations_paired_samples(speeds_dict,relative_angle_dict,experiment,ax,ax2):
	import scipy.signal,scipy.ndimage
	def percentile75(x):
		return numpy.percentile(x,75)
	def percentile25(x):
		return numpy.percentile(x,25)
	treatment_list = []
	treatment_label_list = []
	experiment_titles = {'fish_T':'D. rerio','Gerard_mouse_T':'M. musculus','Gautreau_dicty':'Dictyostelium'}

	study_labels = {'fish_T':'This study','Gerard_mouse':'Gerard et al.','Gautreau_dicty':'Dang et al.'}
	treatment_label_dict = {'control':'Control','rockout':'12 $\mu$M Rockout','Myo1g_KO':'Myo1g KO','Arp_KO':'Arpin KO','Arp_rescue':'Arpin rescue','WT':'WT'}
	treatment_color_dict = {'control':'C0','rockout':'C2','Myo1g_KO':'C2','Arp_KO':'C2','Arp_rescue':'C3','WT':'C0'}
	treatment_color_dict_outlines = {'control':'darkblue','rockout':'darkgreen','Myo1g_KO':'darkgreen','Arp_KO':'darkgreen','Arp_rescue':'Brown','WT':'darkblue'}
	angles_subset = {}
	speeds_subset = {}
	
	angles_se = {}
	speeds_se = {}
	
	n_bootstrap = 500
	
	for treatment in relative_angle_dict[experiment]:
		if 'highfreq' not in treatment:
		
			angles_subset[treatment] = []
			speeds_subset[treatment] = []
			angles_se[treatment] = []
			speeds_se[treatment] = []
			
			for sample in  relative_angle_dict[experiment][treatment]:
				if sample in relative_angle_dict[experiment]['rockout']: ###only include samples in both the rockout and control categories
					for traj_ind in  relative_angle_dict[experiment][treatment][sample]:
						if traj_ind in speeds_dict[experiment][treatment][sample]:
							mean_speed=numpy.mean(speeds_dict[experiment][treatment][sample][traj_ind])
							speed_se = numpy.std(speeds_dict[experiment][treatment][sample][traj_ind])/numpy.sqrt(len(speeds_dict[experiment][treatment][sample][traj_ind]) - 1)
							good_steps = len(speeds_dict[experiment][treatment][sample][traj_ind])
						
							if ~numpy.isnan(mean_speed) and good_steps > 10:
								cosangles = numpy.cos([entry[1] for entry in relative_angle_dict[experiment][treatment][sample][traj_ind]])
								angles_subset[treatment].append( numpy.mean(cosangles))
								angles_se_temp = numpy.std(cosangles)/numpy.sqrt(len(cosangles) - 1)
								speeds_subset[treatment].append( mean_speed )
							
								speeds_se[treatment].append( speed_se )
								angles_se[treatment].append( angles_se_temp )
	
	for treatment in angles_subset:
		if 'highfreq' not in treatment:
			treatment_list.append(treatment)
			treatment_label_list.append(treatment_label_dict[treatment])
			speed_bins = numpy.arange(0,16.1,2)
			#if 'fish' in experiment and 'control' in treatment:
			#	speed_bins = numpy.arange(0,18.1,2)#numpy.percentile(speeds_subset[treatment],numpy.arange(0,101,10))
			#else:
			#	speed_bins = numpy.percentile(speeds_subset[treatment],numpy.arange(0,101,20))
			angles_subset[treatment] = numpy.array(angles_subset[treatment])
			speeds_subset[treatment] = numpy.array(speeds_subset[treatment])
			
			speed_order = numpy.argsort(speeds_subset[treatment])
			angles_ordered = angles_subset[treatment][speed_order]
			speeds_ordered = speeds_subset[treatment][speed_order]
			rolling_median_angles = scipy.ndimage.gaussian_filter(angles_ordered,sigma=10)#scipy.signal.medfilt(angles_ordered,kernel_size=51)
			rolling_median_speeds = scipy.ndimage.gaussian_filter(speeds_ordered,sigma=10)
			
			#ax.plot(rolling_median_speeds,rolling_median_angles,color=treatment_color_dict[treatment],zorder=2)
			
			xlocs,bins,nbins = binned_statistic(speeds_subset[treatment],speeds_subset[treatment],bins=speed_bins)
			mean_ang,bins,nbins = binned_statistic(speeds_subset[treatment],angles_subset[treatment],bins=speed_bins,statistic='median')
			perc75,bins,nbins = binned_statistic(speeds_subset[treatment],angles_subset[treatment],bins=speed_bins,statistic=percentile75)
			perc25,bins,nbins = binned_statistic(speeds_subset[treatment],angles_subset[treatment],bins=speed_bins,statistic=percentile25)
			
			n_cells = len(angles_subset[treatment])
			print(xlocs)
			print(speed_bins)
			print(mean_ang)
			#ax.plot(xlocs[:-1],mean_ang[:-1],color=treatment_color_dict[treatment],zorder=2,linewidth=2)
			#ax.plot(xlocs,perc90,color=treatment_color_dict[treatment],zorder=2,linewidth=.5)
			#ax.plot(xlocs,perc10,color=treatment_color_dict[treatment],zorder=2,linewidth=.5)
			ct_stat_bs = []
			for n in range(n_bootstrap):
				cell_choices = numpy.random.randint(0,high=n_cells,size=n_cells)
				corr_times_temp = angles_subset[treatment][cell_choices]
				speeds_temp = speeds_subset[treatment][cell_choices]
				
				ct_stat,bins,nbins = binned_statistic(speeds_temp,corr_times_temp,bins=speed_bins,statistic='median')
				
				ct_stat_bs.append(ct_stat)
		
		ct_stat_bs = numpy.array(ct_stat_bs)
		
		#ct_stat_med = numpy.percentile(ct_stat_bs,50,axis=0)
		lb = numpy.percentile(ct_stat_bs,5,axis=0)
		ub = numpy.percentile(ct_stat_bs,95,axis=0)
		
		
		if 'control' in treatment:
			xlocs = xlocs[~numpy.isnan(mean_ang)][1:]
			
			lb = lb[~numpy.isnan(mean_ang)][1:]
			ub = ub[~numpy.isnan(mean_ang)][1:]
			mean_ang = mean_ang[~numpy.isnan(mean_ang)][1:]
			ax.errorbar(xlocs,mean_ang,yerr=[mean_ang-lb,ub-mean_ang],marker='o',color=treatment_color_dict_outlines[treatment],zorder=2,markeredgecolor=treatment_color_dict_outlines[treatment])
		else:
			ax.errorbar(xlocs,mean_ang,yerr=[mean_ang-lb,ub-mean_ang],marker='o',color=treatment_color_dict_outlines[treatment],zorder=2,markeredgecolor=treatment_color_dict_outlines[treatment])

			
	for treatment in angles_subset:
		if 'highfreq' not in treatment:
			treatment_list.append(treatment)
			#treatment_label_list.append(treatment_label_dict[treatment])
			#ax.errorbar(speeds_subset[treatment],angles_subset[treatment],xerr=speeds_se[treatment],yerr=angles_se[treatment],marker='o',markersize=2,alpha=.4,elinewidth=.3,linestyle='None')
			ax.errorbar(speeds_subset[treatment],angles_subset[treatment],marker='o',markersize=3,alpha=.7,elinewidth=.3,linestyle='None',color=treatment_color_dict[treatment],zorder=0)

	ax.legend(treatment_label_list)
		
	#ax3.set_xlabel(r'$\langle s \rangle_{cell} (\mu$m/min)')
	ax.set_ylabel(r'$\langle Cos\theta \rangle_{cell}$')
	ax2.set_ylabel(r'$\langle Cos\theta \rangle_{cell}$')
	for treatment in treatment_list:
		#sns.distplot(speeds_subset[treatment],ax=ax2,kde=False,hist_kws={"histtype": "step", "linewidth": 1,"alpha": 1, "color": treatment_color_dict[treatment], "density":True})
		#ax2.hist(speeds_subset[treatment],histtype="step",color=treatment_color_dict[treatment],density=True)
		sns.distplot(speeds_subset[treatment],ax=ax2,kde=False,norm_hist=True,color=treatment_color_dict[treatment],hist_kws={"histtype": "step", "linewidth": 2,"alpha": 1, "color": treatment_color_dict[treatment]})

		#sns.distplot(speeds_subset[treatment],ax=ax2,color=treatment_color_dict[treatment])
	
	ax2.set_xlabel('Cell speed ($\mu$m/min)')
	ax2.set_ylabel('Probability density')
	if experiment=='fish_T' or experiment=='Gerard_mouse_T':
		ax.set_xlim(0,18.5)
		ax2.set_xlim(0,18.5)
	else:
		ax.set_xlim(0,9)
		ax2.set_xlim(0,9)
		
def angle_speed_confounders(speeds_dict,relative_angle_dict,ax,ax2,ax3,ax4):
	def percentile_choice1(x):
		return numpy.percentile(x,90)
	
	speed_bins = numpy.arange(0,16.1,2)
	
	angles_subset = {}
	speeds_subset = {}
	
	angles_se = {}
	speeds_se = {}
	
	experiment = 'fish_T'
	for treatment in relative_angle_dict[experiment]:
		
		if 'control' in treatment and 'highfreq' not in treatment:
		
			angles_subset[treatment] = []
			speeds_subset[treatment] = []
			angles_se[treatment] = []
			speeds_se[treatment] = []
			
			for sample in  relative_angle_dict[experiment][treatment]:
				for traj_ind in  relative_angle_dict[experiment][treatment][sample]:
					if traj_ind in speeds_dict[experiment][treatment][sample]:
						mean_speed=numpy.mean(speeds_dict[experiment][treatment][sample][traj_ind])
						speed_se = numpy.std(speeds_dict[experiment][treatment][sample][traj_ind])/numpy.sqrt(len(speeds_dict[experiment][treatment][sample][traj_ind]) - 1)
						good_steps = len(speeds_dict[experiment][treatment][sample][traj_ind])
						
						if ~numpy.isnan(mean_speed) and good_steps > 10:
							cosangles = numpy.cos([entry[1] for entry in relative_angle_dict[experiment][treatment][sample][traj_ind]])
							angles_subset[treatment].append( numpy.mean(cosangles))
							angles_se_temp = numpy.std(cosangles)/numpy.sqrt(len(cosangles) - 1)
							speeds_subset[treatment].append( mean_speed )
							
							speeds_se[treatment].append( speed_se )
							angles_se[treatment].append( angles_se_temp )
							
	
	for treatment in angles_subset:
		if 'highfreq' not in treatment:
			#treatment_list.append(treatment)
			#treatment_label_list.append(treatment_label_dict[treatment])
			#ax.errorbar(speeds_subset[treatment],angles_subset[treatment],xerr=speeds_se[treatment],yerr=angles_se[treatment],marker='o',markersize=2,alpha=.4,elinewidth=.3,linestyle='None')
			ax.errorbar(speeds_subset[treatment],angles_subset[treatment],marker='o',markersize=2,alpha=.6,elinewidth=.3,linestyle='None')
			angle_stat_fish,binedges,nbins = binned_statistic(speeds_subset[treatment],angles_subset[treatment],bins=speed_bins,statistic=percentile_choice1)
			xlocs,binedges,nbins = binned_statistic(speeds_subset[treatment],speeds_subset[treatment],bins=speed_bins,statistic='mean')
			ax4.plot(xlocs,angle_stat_fish/angle_stat_fish[-1],'-o',markersize=3)
			
	#ax.legend(treatment_label_list)
		
	ax.set_xlabel(r'$\langle s \rangle_{cell} (\mu$m/min)')
	ax.set_ylabel(r'$\langle Cos\theta \rangle_{cell}$')
	ax.set_title('Zebrafish T cells')
	
	
	###Simulate
	
	Sbar = 4.5
	Pbar = 1.25
	ncells = 2000
	v0 = 1
	timescale = 1
	tau = 10
	data_len = 10

	corrts = []
	ms_measured = []
	nt = 2000

	tau_list_base = numpy.array([1,2,4,8,12,16,20,30])
	label_list = []

	alpha = 20
	sigma_noise = 4/numpy.sqrt(alpha)
	sigma_noise_r = sigma_noise/numpy.sqrt(2)# + numpy.tile(sigma_noise*numpy.random.beta(.5,.5,size=ncells),(2,1)).T
	tau_step = alpha
	tmeasure_range = numpy.arange(int(nt/2),nt,1)
	msd_by_tau = {}
	
	S = numpy.tile(1/alpha*4*Sbar*numpy.random.beta(.5,.5,size=ncells),(2,1)).T
	P = alpha*1.4
	
	

	tau_list = alpha*tau_list_base
	trajs = simulate_spc_model.simulate_prw(ncells,nt,S,P,sigma_noise_r)
	polar_trajs = simulate_spc_model.calculate_polar_trajs(trajs,tau_step,tmeasure_range)
	dot_products, disps_ss = simulate_spc_model.calculate_relative_cosangles(trajs,tau_step,tmeasure_range)
	disps_basept = numpy.sqrt( numpy.sum(disps_ss[:-1,:,:]**2,axis=2) )
	corrts = simulate_spc_model.measure_persistence_times_by_traj(polar_trajs,timescale)
	mean_speeds = numpy.mean(disps_basept,axis=0)
	#pt.figure()
	for n in range(ncells):
		if numpy.isnan(mean_speeds[n]):
			print(S[n,:],P[n,:])
			
	mean_corrts = numpy.array([numpy.mean(corrt_cell) for corrt_cell in corrts])
	mean_speeds = numpy.mean(disps_basept,axis=0)
	mean_cosangles = numpy.mean(dot_products,axis=0)
	
	ax2.plot(mean_speeds,mean_cosangles,'o',markersize=2,alpha=.5)
	binned_cosangles1,binedges,nbins = binned_statistic(mean_speeds,mean_cosangles,bins=speed_bins,statistic=percentile_choice1)
	xlocs_speeds,binedges,nbins = binned_statistic(mean_speeds,mean_speeds,bins=speed_bins,statistic='mean')
	ax4.plot(xlocs_speeds,binned_cosangles1/binned_cosangles1[-1],'-o',markersize=3)
	
	ax2.set_xlabel(r'$\langle s \rangle_{cell} (\mu$m/min)')
	ax2.set_ylabel(r'$\langle Cos\theta \rangle_{cell}$')
	ax2.set_title('UPT with noise')
	
	####
	corrts = []
	ms_measured = []
	nt = 10000

	tau_list_base = numpy.array([1,2,4,8,12,16,20,30])
	label_list = []

	alpha = 20
	sigma_noise = 4/numpy.sqrt(alpha)
	sigma_noise_r = sigma_noise/numpy.sqrt(2)# + numpy.tile(sigma_noise*numpy.random.beta(.5,.5,size=ncells),(2,1)).T
	tau_step = alpha
	tmeasure_range = numpy.arange(int(nt/2),nt,1)
	msd_by_tau = {}

	S = numpy.tile(1/alpha*4*Sbar*numpy.random.beta(.5,.5,size=ncells),(2,1)).T
	P = numpy.maximum(.5,numpy.tile(alpha*1.1*Pbar*numpy.random.beta(.5,.5,size=ncells),(2,1)).T)

	tau_list = alpha*tau_list_base
	trajs = simulate_spc_model.simulate_prw(ncells,nt,S,P,0)
	polar_trajs = simulate_spc_model.calculate_polar_trajs(trajs,tau_step,tmeasure_range)
	dot_products, disps_ss = simulate_spc_model.calculate_relative_cosangles(trajs,tau_step,tmeasure_range)
	disps_basept = numpy.sqrt( numpy.sum(disps_ss[:-1,:,:]**2,axis=2) )
	corrts = simulate_spc_model.measure_persistence_times_by_traj(polar_trajs,timescale)
	mean_speeds = numpy.mean(disps_basept,axis=0)
	#pt.figure()
	for n in range(ncells):
		if numpy.isnan(mean_speeds[n]):
			print(S[n,:],P[n,:])
			
	mean_corrts = numpy.array([numpy.mean(corrt_cell) for corrt_cell in corrts])
	mean_speeds = numpy.mean(disps_basept,axis=0)
	mean_cosangles = numpy.mean(dot_products,axis=0)
	
	ax3.plot(mean_speeds,mean_cosangles,'o',markersize=2,alpha=.5)
	binned_cosangles1,binedges,nbins = binned_statistic(mean_speeds,mean_cosangles,bins=speed_bins,statistic=percentile_choice1)
	xlocs_speeds,binedges,nbins = binned_statistic(mean_speeds,mean_speeds,bins=speed_bins,statistic='mean')
	ax4.plot(xlocs_speeds,binned_cosangles1/binned_cosangles1[-1],'-o',markersize=3)
	
	ax3.set_xlabel(r'$\langle s \rangle_{cell} (\mu$m/min)')
	ax3.set_ylabel(r'$\langle Cos\theta \rangle_{cell}$')
	ax3.set_title('UPT, uncorrelated S and P')
	
	####
	S = numpy.tile(1/alpha*4*Sbar*numpy.random.beta(.5,.5,size=ncells),(2,1)).T
	P = numpy.maximum(.5,numpy.tile(alpha*1.1*Pbar*numpy.random.beta(.5,.5,size=ncells),(2,1)).T)

	tau_list = alpha*tau_list_base
	trajs = simulate_spc_model.simulate_prw(ncells,nt,S,P,sigma_noise_r)
	polar_trajs = simulate_spc_model.calculate_polar_trajs(trajs,tau_step,tmeasure_range)
	dot_products, disps_ss = simulate_spc_model.calculate_relative_cosangles(trajs,tau_step,tmeasure_range)
	disps_basept = numpy.sqrt( numpy.sum(disps_ss[:-1,:,:]**2,axis=2) )
	corrts = simulate_spc_model.measure_persistence_times_by_traj(polar_trajs,timescale)
	mean_speeds = numpy.mean(disps_basept,axis=0)
	#pt.figure()
	for n in range(ncells):
		if numpy.isnan(mean_speeds[n]):
			print(S[n,:],P[n,:])
			
	mean_corrts = numpy.array([numpy.mean(corrt_cell) for corrt_cell in corrts])
	mean_speeds = numpy.mean(disps_basept,axis=0)
	mean_cosangles = numpy.mean(dot_products,axis=0)
	
	
	binned_cosangles1,binedges,nbins = binned_statistic(mean_speeds,mean_cosangles,bins=speed_bins,statistic=percentile_choice1)
	xlocs_speeds,binedges,nbins = binned_statistic(mean_speeds,mean_speeds,bins=speed_bins,statistic='mean')
	ax4.plot(xlocs_speeds,binned_cosangles1/binned_cosangles1[-1],'-o',markersize=3)
	
	ax4.set_xlabel(r'$\langle s \rangle_{cell} (\mu$m/min) ')
	ax4.set_ylabel(r'$\langle Cos\theta \rangle_{cell},$90$^{th}$ percentile (scaled)')
	ax4.legend(['Data','UPT with noise (1)','UPT, uncorrelated S and P (2)','(1) and (2) combined'],fontsize=8)

###Note: fcn below not currently in use	
def persistence_length_dists(corr_lengths_dict, speeds_dict, ax):
	
	experiment_list = ['fish_T']
	experiment_titles = ['D. rerio','M. musculus','Dictyostelium']

	study_labels = ['This study','Gerard et al.','Dang et al.']
	treatment_label_dict = {'control':'Control','rockout':'12 $\mu$M Rockout','Myo1g_KO':'Myo1g KO','Arp_KO':'Arpin KO','Arp_rescue':'Arpin rescue','WT':'WT'}
	experiment_ind = 0

	traj_list = {}
	mean_speed_list = {}

	ttot = 400
	hz_frequencies = 2*numpy.pi*(1/45)*numpy.arange(int(ttot/2))*1/ttot
	
	###Collect mean speeds to compute quintile speed classes
	
	mean_speed_list_all = []
	
	tmin = 20
	
	for experiment in corr_lengths_dict:
		if 'fish' in experiment:
			for treatment in corr_lengths_dict[experiment]:
				if 'control' in treatment and 'highfreq' not in treatment:
					
					for sample in corr_lengths_dict[experiment][treatment]:
						
						for traj_ind in corr_lengths_dict[experiment][treatment][sample]:
							
							if traj_ind in speeds_dict[experiment][treatment][sample]:
								mean_speed=numpy.mean(speeds_dict[experiment][treatment][sample][traj_ind])
								if ~numpy.isnan(mean_speed) and len(speeds_dict[experiment][treatment][sample][traj_ind]) >= tmin:
									mean_speed_list_all.append(mean_speed)
	
	speed_classes = numpy.percentile(mean_speed_list_all,numpy.arange(0,101,20))
	
	handles = []
	corr_lengths_by_speed_class = {}
	for experiment in experiment_list:
		
		treatment_list = []
		for treatment in corr_lengths_dict[experiment]:
			if 'control' in treatment and 'highfreq' not in treatment:
				for sample in corr_lengths_dict[experiment][treatment]:
					
					for traj_ind in corr_lengths_dict[experiment][treatment][sample]:
						mean_speed=numpy.mean(speeds_dict[experiment][treatment][sample][traj_ind])
						if len(speeds_dict[experiment][treatment][sample][traj_ind]) > tmin and ~numpy.isnan(mean_speed):
						
							speed_class = numpy.argmax( mean_speed - speed_classes < 0 ) - 1
									
							if speed_class < 0:
								speed_class = len(speed_classes) - 1
							
							if speed_class not in traj_list:
								traj_list[speed_class] = []
								mean_speed_list[speed_class] = []
								
							traj_list[speed_class].append((sample,traj_ind))
							mean_speed_list[speed_class].append(mean_speed)
							if speed_class not in corr_lengths_by_speed_class:
								corr_lengths_by_speed_class[speed_class] = []
							corr_lengths_by_speed_class[speed_class].extend(corr_lengths_dict[experiment][treatment][sample][traj_ind]/mean_speed)
	
	speed_label_list = []
	handles = []
	for speed_class in range(5):
		
		speed_label_list.append( str(numpy.round(numpy.mean(mean_speed_list[speed_class]),1)) + r' $\frac{\mu m}{min}$')
		dist_bins = numpy.percentile(corr_lengths_by_speed_class[speed_class],numpy.arange(0,101,10))
		print(dist_bins)
		distribution,binedges = numpy.histogram(corr_lengths_by_speed_class[speed_class],bins=dist_bins,density=True)
		binlocs,binedges,nbins = binned_statistic(corr_lengths_by_speed_class[speed_class],corr_lengths_by_speed_class[speed_class],bins=dist_bins)
		print(binlocs,distribution)
		handle=ax.loglog(binlocs,distribution,'-o')
		handles.append(handle)
	ax.set_xlabel('Bout length/speed')
	ax.set_ylabel('Probability')
	lgnd = ax.legend(handles,speed_label_list,fontsize=8,loc="upper right",handletextpad=.2,frameon=False)
						