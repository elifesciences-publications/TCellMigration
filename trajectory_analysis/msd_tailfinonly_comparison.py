import trajectory_analysis_functions_09192019
import numpy
import matplotlib.pylab as pt
import seaborn as sns
import matplotlib.gridspec as gridspec
import matplotlib
import data_import_and_basic_calculations_09202019
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit

####This script is a top-level wrapper for paper figures. It first imports the trajectory data that is used in most figure panels; these data structures are passed to subfunctions in the figure_panel_functions_11242019.py library for plotting.
####This script takes axes returned by figure_panel_functions_11242019 and arranges them into multiplanel figures.

master_trajectory_dict, trajectory_dict_polar, trajectory_dict_polar_interp, speeds_dict, corr_times_dict, corr_times_shallow_dict, corr_lengths_dict, step_angle_coupling_dict = data_import_and_basic_calculations_09202019.import_data_and_measure( 'master_trajectory_file_all_experiments.txt' )


	
def MSD_overall(speeds_dict, drift_corrected_traj_dict, ax, inset_ax):
	
    msds_by_tau = {}
    mean_speeds = {}
    indexes_by_tau = {}

    taus = [45,90,135,180,225,270,315,360,450,540,630,720,810,900,1080,1080+180*1,1080+180*2,1080+180*3,1080+180*4,1080+180*6,1080+180*8,1080+180*10,1080+180*12,1080+180*14,1080+180*16,1080+180*18,1080+180*20,1080+180*22,1080+180*24]

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
                            #msds_by_traj = trajectory_analysis_functions_09192019.calculate_MSDs(drift_corrected_traj_dict[experiment][treatment][sample], tau, overlap=True)
                            msds_by_traj = trajectory_analysis_functions_09192019.calculate_MSDs_alt(drift_corrected_traj_dict[experiment][treatment][sample], tau)
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
    lb = numpy.percentile(mean_msds_bs_list,2.5,axis=0)
    ub = numpy.percentile(mean_msds_bs_list,97.5,axis=0)

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

    #wlc_fit = curve_fit(WLC_MSD,taus,med_stat,p0=[2,1,1],sigma=ub-lb)[0]

    params_early = numpy.polyfit(numpy.log10(early_t),numpy.log10(early_msd),deg=1,w=1/numpy.log10(ub_early/lb_early))
    params_late = numpy.polyfit(numpy.log10(late_t),numpy.log10(late_msd),deg=1,w=1/numpy.log10(ub_late/lb_late))

    ax.errorbar(taus,med_stat,yerr=[med_stat-lb,ub-med_stat],marker='o',linestyle='None',zorder=1,markersize=5,alpha=.8,label='All other control samples')

    print(params_early)
    print(params_late)

    #print(wlc_fit)

    #handle=ax.plot(taus, WLC_MSD(taus,*wlc_fit),'k',zorder=2)


    ax.plot(early_t, 1.5*numpy.power(10,params_early[0]*numpy.log10(early_t) + params_early[1]), 'k--',zorder=2)
    yvals = 1.5*numpy.power(10,params_early[0]*numpy.log10(early_t) + params_early[1])

    ax.plot([early_t[2],early_t[2]],[yvals[2],yvals[3]],'k')
    ax.plot([early_t[2],early_t[3]],[yvals[3],yvals[3]],'k')


    ax.text(1.6,300,str(round(params_early[0],1)),horizontalalignment='center')



    ax.set_ylabel(r'$\langle MSD(\tau) \rangle$ ($\mu m^2$)')
    ax.set_xlabel(r'$\tau$ (min)')

    inset_ax.errorbar(early_t,early_msd,yerr=[early_msd-lb_early,ub_early-early_msd],marker='o',markersize=2,elinewidth=.3,linewidth=.5)
    inset_ax.tick_params(labelsize=6,length=1)
    inset_ax.set_ylabel(r'$\langle MSD(\tau) \rangle$',fontsize=8)
    inset_ax.set_xlabel(r'$\tau$ (min)',fontsize=8)
    #ax.legend(handle,['PRW Fit'])

def MSD_overall_add_highfreq(speeds_dict, drift_corrected_traj_dict, ax, inset_ax):
	
    msds_by_tau = {}
    mean_speeds = {}
    indexes_by_tau = {}

    taus = numpy.hstack((numpy.arange(48,12*48+1,48),numpy.arange(14*48,24*48+1,2*48),numpy.arange(28*48+1,60*48+1,4*48),numpy.arange(64*48,120*48+1,8*48)))
                        

    for experiment in speeds_dict:
        if 'fish' in experiment:
            for treatment in speeds_dict[experiment]:
                if 'control' in treatment and 'highfreq' in treatment:

                    for tau in taus:
                        msds_by_tau[tau] = {}
                        mean_speeds[tau] = {}
                        indexes_by_tau[tau] = []
                        for sample in speeds_dict[experiment][treatment]:
                            #print('about to calculate msds')
                            #msds_by_traj = trajectory_analysis_functions_09192019.calculate_MSDs(drift_corrected_traj_dict[experiment][treatment][sample], tau, overlap=True)
                            msds_by_traj = trajectory_analysis_functions_09192019.calculate_MSDs_alt(drift_corrected_traj_dict[experiment][treatment][sample], tau)
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
    lb = numpy.percentile(mean_msds_bs_list,2.5,axis=0)
    ub = numpy.percentile(mean_msds_bs_list,97.5,axis=0)

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

    #wlc_fit = curve_fit(WLC_MSD,taus,med_stat,p0=[2,1,1],sigma=ub-lb)[0]

    params_early = numpy.polyfit(numpy.log10(early_t),numpy.log10(early_msd),deg=1,w=1/numpy.log10(ub_early/lb_early))
    params_late = numpy.polyfit(numpy.log10(late_t),numpy.log10(late_msd),deg=1,w=1/numpy.log10(ub_late/lb_late))

    ax.errorbar(taus,med_stat,yerr=[med_stat-lb,ub-med_stat],marker='o',linestyle='None',zorder=1,markersize=5,alpha=.8,label='Tail fin only sample')

    print(params_early)
    print(params_late)

    #print(wlc_fit)

    #handle=ax.plot(taus, WLC_MSD(taus,*wlc_fit),'k',zorder=2)


    #ax.plot(early_t, 1.5*numpy.power(10,params_early[0]*numpy.log10(early_t) + params_early[1]), 'k--',zorder=2)
    yvals = 1.5*numpy.power(10,params_early[0]*numpy.log10(early_t) + params_early[1])

    #ax.plot([early_t[2],early_t[2]],[yvals[2],yvals[3]],'k')
    #ax.plot([early_t[2],early_t[3]],[yvals[3],yvals[3]],'k')


    #ax.text(1.6,300,str(round(params_early[0],1)),horizontalalignment='center')



    ax.set_ylabel(r'$\langle MSD(\tau) \rangle$ ($\mu m^2$)')
    ax.set_xlabel(r'$\tau$ (min)')

    inset_ax.errorbar(early_t,early_msd,yerr=[early_msd-lb_early,ub_early-early_msd],marker='o',markersize=2,elinewidth=.3,linewidth=.5)
    inset_ax.tick_params(labelsize=6,length=1)
    inset_ax.set_ylabel(r'$\langle MSD(\tau) \rangle$',fontsize=8)
    inset_ax.set_xlabel(r'$\tau$ (min)',fontsize=8)
    #ax.legend(handle,['PRW Fit'])
    
fig = pt.figure(figsize=(4.5,4.5))
###first row
wstep = 1
hstep = 1
p_width = .9
p_height = .9
rect_locations = [[0,0,1,1],[.6*wstep,.12*hstep,p_width/2.5,p_height/2.5]]

####Create the requisite number of axis objects

my_axes = []
for rect_loc in rect_locations:

    ax = fig.add_axes(rect_loc)
    my_axes.append(ax)

MSD_overall(speeds_dict, master_trajectory_dict, ax=my_axes[0],inset_ax=my_axes[1])
MSD_overall_add_highfreq(speeds_dict, master_trajectory_dict, ax=my_axes[0],inset_ax=my_axes[1])
my_axes[0].legend(loc="upper left")
pt.savefig('/Users/ejerison/Dropbox/imaging_data/figures/revision_figures/AppendixFig2_extrasampcomp_v2.pdf',bbox_inches='tight')