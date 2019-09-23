import numpy
import matplotlib.pylab as pt
from scipy.stats import binned_statistic
import simulate_spc_model
import data_import_and_basic_calculations

master_trajectory_dict, trajectory_dict_polar, trajectory_dict_polar_interp, speeds_dict, corr_times_dict, corr_lengths_dict, relative_angle_dict = data_import_and_basic_calculations.import_data_and_measure( 'master_trajectory_file_all_experiments.txt' )

fig,((ax1,ax2,ax3),(ax4,ax5,ax6)) = pt.subplots(2,3,figsize=(7.5,6.5),sharey='all')

###Measure the list of mean speeds and persistence times in the data; also measure <cos\theta>_cell

def percentile_choice1(x):
	return numpy.percentile(x,90)
	
def std_errs(x):
	return numpy.std(x)/numpy.sqrt(len(x)-1)

angles_subset = []
speeds_subset = []
corrts_data = []

experiment = 'fish_T'
for treatment in relative_angle_dict[experiment]:
	
	if 'control' in treatment and 'highfreq' not in treatment:
		
		for sample in  relative_angle_dict[experiment][treatment]:
			for traj_ind in  relative_angle_dict[experiment][treatment][sample]:
				if traj_ind in speeds_dict[experiment][treatment][sample] and traj_ind in corr_times_dict[experiment][treatment][sample]:
					mean_speed=numpy.mean(speeds_dict[experiment][treatment][sample][traj_ind])
					speed_se = numpy.std(speeds_dict[experiment][treatment][sample][traj_ind])/numpy.sqrt(len(speeds_dict[experiment][treatment][sample][traj_ind]) - 1)
					good_steps = len(speeds_dict[experiment][treatment][sample][traj_ind])
					
					if ~numpy.isnan(mean_speed) and good_steps > 10:
						cosangles = numpy.cos([entry[1] for entry in relative_angle_dict[experiment][treatment][sample][traj_ind]])
						angles_subset.append( numpy.mean(cosangles))
						speeds_subset.append( mean_speed )
						corrts_data.append(numpy.mean(corr_times_dict[experiment][treatment][sample][traj_ind]))
						
speeds_data = numpy.array(speeds_subset)
corrts_data = numpy.array(corrts_data)
angles_subset = numpy.array(angles_subset)
#ax1.plot(speeds_data,angles_subset,linestyle='None',marker='o',markersize=3,alpha=.6)
print(numpy.mean(corrts_data/60.),numpy.mean(speeds_data))
speed_bins = numpy.percentile(speeds_data,numpy.arange(0,101,10))

binned_correlation_times,binedges,nbins = binned_statistic(speeds_data,corrts_data/60.,bins=speed_bins)
xlocs,binedges,nbins = binned_statistic(speeds_data,speeds_data,bins=speed_bins)
errs,binedges,nbins = binned_statistic(speeds_data,corrts_data/60.,bins=speed_bins,statistic=std_errs)
#ax1.plot(speeds_data,corrts_data/60.,linestyle='None',marker='o',markersize=2,alpha=.6,zorder=0)
ax1.errorbar(xlocs,binned_correlation_times,yerr=2*errs,marker='o',markersize=5,)

###Fit speed v corrt
params = numpy.polyfit(speeds_data, corrts_data/60.,  deg=1)
print(params)
model_P = (speeds_data*params[0] + params[1])



###Simulations
ncells = len(speeds_data)
timescale = 1
tau_step = 20
nt = 2000
tmeasure_range = numpy.arange(int(nt/2),nt,1)

###Constant P; finite length trajectories; distribution of S (UPT model)

S = 1/tau_step*1/.85*numpy.tile(speeds_data,(2,1)).T
P = numpy.mean(corrts_data)/60*tau_step
sigma_noise = 0
trajs = simulate_spc_model.simulate_prw(ncells,nt,S,P,sigma_noise)
polar_trajs = simulate_spc_model.calculate_polar_trajs(trajs,tau_step,tmeasure_range)
dot_products, disps_ss = simulate_spc_model.calculate_relative_cosangles(trajs,tau_step,tmeasure_range)
disps_basept = numpy.sqrt( numpy.sum(disps_ss[:-1,:,:]**2,axis=2) )
corrts = simulate_spc_model.measure_persistence_times_by_traj(polar_trajs,timescale)

mean_speeds = numpy.mean(disps_basept,axis=0)*4/3
mean_corrts = numpy.array([numpy.mean(corrt_cell) for corrt_cell in corrts])*.75
mean_cosangles = numpy.mean(dot_products,axis=0)
print(numpy.mean(mean_corrts),numpy.mean(mean_speeds))

#ax6.set_ylim(.25,2.2)
#ax3.plot(mean_speeds,mean_cosangles,linestyle='None',marker='o',markersize=3,alpha=.6)
binned_correlation_times,binedges,nbins = binned_statistic(mean_speeds,mean_corrts,bins=speed_bins)
xlocs,binedges,nbins = binned_statistic(mean_speeds,mean_speeds,bins=speed_bins)
errs,binedges,nbins = binned_statistic(mean_speeds,mean_corrts,bins=speed_bins,statistic=std_errs)
#ax3.plot(mean_speeds,mean_corrts,linestyle='None',marker='o',markersize=2,alpha=.6,zorder=0)
ax3.errorbar(xlocs,binned_correlation_times,yerr=2*errs,marker='o',markersize=5,color='C2')

###Constant P; finite length trajectories; distribution of S; noise

S = 1/tau_step*1/.85*numpy.tile(speeds_data,(2,1)).T
P = numpy.mean(corrts_data)/60*tau_step
sigma_noise = 2/numpy.sqrt(tau_step)
trajs = simulate_spc_model.simulate_prw(ncells,nt,S,P,sigma_noise)
polar_trajs = simulate_spc_model.calculate_polar_trajs(trajs,tau_step,tmeasure_range)
dot_products, disps_ss = simulate_spc_model.calculate_relative_cosangles(trajs,tau_step,tmeasure_range)
disps_basept = numpy.sqrt( numpy.sum(disps_ss[:-1,:,:]**2,axis=2) )
corrts = simulate_spc_model.measure_persistence_times_by_traj(polar_trajs,timescale)

mean_speeds = numpy.mean(disps_basept,axis=0)*4/3
mean_corrts = numpy.array([numpy.mean(corrt_cell) for corrt_cell in corrts])*.75
mean_cosangles = numpy.mean(dot_products,axis=0)

#ax3.plot(mean_speeds,mean_cosangles,linestyle='None',marker='o',markersize=3,alpha=.6)
binned_correlation_times,binedges,nbins = binned_statistic(mean_speeds,mean_corrts,bins=speed_bins)
xlocs,binedges,nbins = binned_statistic(mean_speeds,mean_speeds,bins=speed_bins)
errs,binedges,nbins = binned_statistic(mean_speeds,mean_corrts,bins=speed_bins,statistic=std_errs)
#ax3.plot(mean_speeds,mean_corrts,linestyle='None',marker='o',markersize=2,alpha=.6,zorder=0)
ax4.errorbar(xlocs,binned_correlation_times,yerr=2*errs,marker='o',markersize=5,color='C2')

###Scrambled S and P
permuted_corrts = numpy.random.permutation(model_P)
#permuted_corrts = numpy.random.permutation(corrts_data)
S = 1/tau_step*1/.85*numpy.tile(speeds_data,(2,1)).T
P = numpy.tile(permuted_corrts,(2,1)).T*tau_step
sigma_noise = 0

trajs = simulate_spc_model.simulate_prw(ncells,nt,S,P,sigma_noise)
polar_trajs = simulate_spc_model.calculate_polar_trajs(trajs,tau_step,tmeasure_range)
dot_products, disps_ss = simulate_spc_model.calculate_relative_cosangles(trajs,tau_step,tmeasure_range)
disps_basept = numpy.sqrt( numpy.sum(disps_ss[:-1,:,:]**2,axis=2) )
corrts = simulate_spc_model.measure_persistence_times_by_traj(polar_trajs,timescale)
print(numpy.mean(mean_corrts),numpy.mean(mean_speeds))
mean_speeds = numpy.mean(disps_basept,axis=0)*4/3
mean_corrts = numpy.array([numpy.mean(corrt_cell) for corrt_cell in corrts])*.75
mean_cosangles = numpy.mean(dot_products,axis=0)

#ax5.plot(mean_speeds,mean_cosangles,linestyle='None',marker='o',markersize=3,alpha=.6)
binned_correlation_times,binedges,nbins = binned_statistic(mean_speeds,mean_corrts,bins=speed_bins)
xlocs,binedges,nbins = binned_statistic(mean_speeds,mean_speeds,bins=speed_bins)
errs,binedges,nbins = binned_statistic(mean_speeds,mean_corrts,bins=speed_bins,statistic=std_errs)
#ax5.plot(mean_speeds,mean_corrts,linestyle='None',marker='o',markersize=2,alpha=.6,zorder=0)
ax5.errorbar(xlocs,binned_correlation_times,yerr=2*errs,marker='o',markersize=5,color='C2')

###Scrambled S and P + noise

permuted_corrts = numpy.random.permutation(model_P)
S = 1/tau_step*1/.85*numpy.tile(speeds_data,(2,1)).T
P = numpy.tile(permuted_corrts,(2,1)).T*tau_step
sigma_noise = 2/numpy.sqrt(tau_step)

trajs = simulate_spc_model.simulate_prw(ncells,nt,S,P,sigma_noise)
polar_trajs = simulate_spc_model.calculate_polar_trajs(trajs,tau_step,tmeasure_range)
dot_products, disps_ss = simulate_spc_model.calculate_relative_cosangles(trajs,tau_step,tmeasure_range)
disps_basept = numpy.sqrt( numpy.sum(disps_ss[:-1,:,:]**2,axis=2) )
corrts = simulate_spc_model.measure_persistence_times_by_traj(polar_trajs,timescale)
print(numpy.mean(mean_corrts),numpy.mean(mean_speeds))
mean_speeds = numpy.mean(disps_basept,axis=0)*4/3
mean_corrts = numpy.array([numpy.mean(corrt_cell) for corrt_cell in corrts])*.75
mean_cosangles = numpy.mean(dot_products,axis=0)

#ax5.plot(mean_speeds,mean_cosangles,linestyle='None',marker='o',markersize=3,alpha=.6)
binned_correlation_times,binedges,nbins = binned_statistic(mean_speeds,mean_corrts,bins=speed_bins)
xlocs,binedges,nbins = binned_statistic(mean_speeds,mean_speeds,bins=speed_bins)
errs,binedges,nbins = binned_statistic(mean_speeds,mean_corrts,bins=speed_bins,statistic=std_errs)
#ax6.plot(mean_speeds,mean_corrts,linestyle='None',marker='o',markersize=2,alpha=.6,zorder=0)
ax6.errorbar(xlocs,binned_correlation_times,yerr=2*errs,marker='o',markersize=5,color='C2')

###'actual' S,S-dependent P,noise
S = 1/tau_step*1/.85*numpy.tile(speeds_data,(2,1)).T
P = numpy.tile(model_P,(2,1)).T*tau_step
sigma_noise = 2/numpy.sqrt(tau_step)

trajs = simulate_spc_model.simulate_prw(ncells,nt,S,P,sigma_noise)
polar_trajs = simulate_spc_model.calculate_polar_trajs(trajs,tau_step,tmeasure_range)
dot_products, disps_ss = simulate_spc_model.calculate_relative_cosangles(trajs,tau_step,tmeasure_range)
disps_basept = numpy.sqrt( numpy.sum(disps_ss[:-1,:,:]**2,axis=2) )
corrts = simulate_spc_model.measure_persistence_times_by_traj(polar_trajs,timescale)

mean_speeds = numpy.mean(disps_basept,axis=0)*4/3
mean_corrts = numpy.array([numpy.mean(corrt_cell) for corrt_cell in corrts])*.75
mean_cosangles = numpy.mean(dot_products,axis=0)

ax1.set_ylim(.25,2.5)
ax3.set_ylim(.25,2.5)
ax3.set_ylim(.25,2.5)
ax5.set_ylim(.25,2.5)
ax6.set_ylim(.25,2.5)
ax1.set_xlim(0,15)
ax3.set_xlim(0,15)
ax3.set_xlim(0,15)
ax5.set_xlim(0,15)
ax6.set_xlim(0,15)
ax2.set_xlim(0,15)

#ax6.plot(mean_speeds,mean_cosangles,linestyle='None',marker='o',markersize=3,alpha=.6)
binned_correlation_times,binedges,nbins = binned_statistic(mean_speeds,mean_corrts,bins=speed_bins)
xlocs,binedges,nbins = binned_statistic(mean_speeds,mean_speeds,bins=speed_bins)
errs,binedges,nbins = binned_statistic(mean_speeds,mean_corrts,bins=speed_bins,statistic=std_errs)
##ax2.plot(mean_speeds,mean_corrts,linestyle='None',marker='o',markersize=2,alpha=.6,zorder=0)
ax2.errorbar(xlocs,binned_correlation_times,yerr=2*errs,marker='o',markersize=5)
#ax6.plot(xlocs,binned_correlation_times,marker='o',markersize=4)
ax1.set_xlabel(r'Cell speed ($\mu$m/min)')
ax1.set_ylabel('Persistence time (min)')
ax4.set_ylabel('Persistence time (min)')
ax1.set_title('Zebrafish T cells',fontsize=10)
ax3.set_xlabel(r'Cell speed ($\mu$m/min)')
ax4.set_xlabel(r'Cell speed ($\mu$m/min)')
ax5.set_xlabel(r'Cell speed ($\mu$m/min)')
ax6.set_xlabel(r'Cell speed ($\mu$m/min)')
ax2.set_xlabel(r'Cell speed ($\mu$m/min)')
ax3.set_title('UPT',fontsize=10)
ax4.set_title('UPT with noise',fontsize=10)
ax5.set_title('S and P uncorrelated',fontsize=10)
ax6.set_title('S and P uncorrelated with noise',fontsize=10)
ax2.set_title('SPC with noise',fontsize=10)
pt.tight_layout()
pt.savefig('/mnt/c/Users/ejerison/Dropbox/imaging_data/figures/empirical_sims_v1.pdf')