import trajectory_analysis_functions_09192019
import numpy
import matplotlib.pylab as pt
import seaborn as sns
import matplotlib.gridspec as gridspec
import matplotlib
import data_import_and_basic_calculations_09202019
import figure_panel_functions_09192019
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import binned_statistic, spearmanr
import simulate_spc_model
from scipy.optimize import curve_fit
import pandas

def calculate_rankcorr(mspeed_arr):
	
	rho,p = spearmanr(mspeed_arr,axis=0)
	avg_vals = []
	
	for i in range(rho.shape[0]-1):
		avg_vals.append( numpy.mean( numpy.diagonal(rho,offset=i+1) ) )
	return avg_vals
	
	
master_trajectory_dict, trajectory_dict_polar, trajectory_dict_polar_interp, speeds_dict, corr_times_dict, corr_lengths_dict, step_angle_coupling_dict = data_import_and_basic_calculations_09202019.import_data_and_measure( 'master_trajectory_file_all_experiments.txt' )
master_trajectory_dict_subsamp, trajectory_dict_polar_subsamp, trajectory_dict_polar_interp_subsamp, speeds_dict_subsamp, corr_times_dict_subsamp, corr_lengths_dict_subsamp, step_angle_coupling_dict_subsamp = data_import_and_basic_calculations_09202019.import_data_and_measure_2x_subsampled( 'master_trajectory_file_all_experiments.txt' )

#####Focusing first on the 20 min interval; plot rank correlation (bootstrapped over cells) as well as expectation based on random permutation.
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

pt.figure(figsize=(4,3))
pt.errorbar((time_chunks4[:-1]+45)/60., corr_vals_50, yerr=[corr_vals_50-corr_vals_5,corr_vals_95-corr_vals_50], marker='o',markersize=4)
pt.errorbar((time_chunks4[:-1]+45)/60., null_vals_50, yerr=[null_vals_50-null_vals_5,null_vals_95-null_vals_50], marker='o',markersize=4)
pt.xlabel('Time (min)')
pt.ylabel(r'Speed rank correlation ($\rho$)')
pt.legend(['Trajectories','Null (permuted)'])
pt.savefig('/Users/ejerison/Dropbox/imaging_data/spearman_corr_bs.pdf',bbox_inches='tight')

pt.figure(figsize=(4,3))
ax = pt.gca()
#ax.set_yscale('log')
pt.errorbar((time_chunks4[:-1]+45)/60., corr_vals_50, yerr=[corr_vals_50-corr_vals_5,corr_vals_95-corr_vals_50], marker='o',markersize=4)
pt.errorbar((time_chunks4[:-1]+45)/60., null_vals_50, yerr=[null_vals_50-null_vals_5,null_vals_95-null_vals_50], marker='o',markersize=4)
pt.xlabel('Time (min)')
pt.ylabel(r'Speed rank correlation ($\rho$)')
pt.legend(['Trajectories','Null (permuted)'])
pt.savefig('/Users/ejerison/Dropbox/imaging_data/spearman_corr_semilog.pdf',bbox_inches='tight')

###Fit an exponential decay slope to the first 4 points:
def lf(x,a,b):
	return b*numpy.exp(-1*x/a)
popt,pcov=curve_fit(lf,(time_chunks4[:-2]-time_chunks4[0])/3600.,corr_vals_50[:-1],p0=[.1,.8],sigma=corr_vals_95[:-1]-corr_vals_5[:-1])
print(popt)
pt.figure(figsize=(6,6))
df = pandas.DataFrame(speed_arr4,columns=['0','20','40','60','80','100'])
g=sns.pairplot(df,plot_kws=dict(s=5, edgecolor="b", alpha=.6))
pt.savefig('/Users/ejerison/Dropbox/imaging_data/speed_corr_mat_20min.pdf')