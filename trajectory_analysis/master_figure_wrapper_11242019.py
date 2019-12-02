import trajectory_analysis_functions_09192019
import numpy
import matplotlib.pylab as pt
import seaborn as sns
import matplotlib.gridspec as gridspec
import matplotlib
import data_import_and_basic_calculations_09202019
import figure_panel_functions_11242019
from mpl_toolkits.mplot3d import Axes3D

####This script is a top-level wrapper for paper figures. It first imports the trajectory data that is used in most figure panels; these data structures are passed to subfunctions in the figure_panel_functions_11242019.py library for plotting.
####This script takes axes returned by figure_panel_functions_11242019 and arranges them into multiplanel figures.

master_trajectory_dict, trajectory_dict_polar, trajectory_dict_polar_interp, speeds_dict, corr_times_dict, corr_lengths_dict, step_angle_coupling_dict = data_import_and_basic_calculations_09202019.import_data_and_measure( 'master_trajectory_file_all_experiments.txt' )
master_trajectory_dict_subsamp, trajectory_dict_polar_subsamp, trajectory_dict_polar_interp_subsamp, speeds_dict_subsamp, corr_times_dict_subsamp, corr_lengths_dict_subsamp, step_angle_coupling_dict_subsamp = data_import_and_basic_calculations_09202019.import_data_and_measure_2x_subsampled( 'master_trajectory_file_all_experiments.txt' )

#for entry in corr_times_dict:
#	print(entry,corr_times_dict[entry].keys())

def MSD_PSD_boutx_pooled_panels(master_trajectory_dict, trajectory_dict_polar_interp, step_angle_coupling_dict):
	
	#This is a wrapper for panels C-E in Figure 1
	
	fig = pt.figure(figsize = (8,2))
	
	####Specify [left,bottom,width,height] for each sub-panel
	
	pad = .26
	
	ncol = 3
	nrows = 1
	
	wstep = 1/ncol
	hstep = 1/nrows
	
	p_width = 1/ncol - 1/ncol*pad
	p_height = 1/nrows - 1/nrows*0
	
	###first row
	rect_locations = [[0,2*hstep,p_width,p_height],[.48*wstep,2*hstep+.2*hstep,p_width/3,p_height/3],[wstep,2*hstep,p_width,p_height],[2*wstep,2*hstep,p_width,p_height]]
	
	####Create the requisite number of axis objects
	
	my_axes = []
	for rect_loc in rect_locations:
		
		ax = fig.add_axes(rect_loc)
		my_axes.append(ax)
	
	print('Starting MSD calculation')
	figure_panel_functions_11242019.MSD_overall(speeds_dict, master_trajectory_dict, ax=my_axes[0],inset_ax=my_axes[1])
	print('Done calculating overall MSDs')
	figure_panel_functions_11242019.PSDs_overall(corr_times_dict, trajectory_dict_polar_interp, ax=my_axes[2])
	print('Done calculating overall PSDs')
	figure_panel_functions_11242019.boutx_distribution( speeds_dict,trajectory_dict_polar_interp,ax=my_axes[3])
	print('Done calculating boutx')
	
	#pt.savefig('/mnt/c/Users/ejerison/Dropbox/imaging_data/figures/paper_figures/Figure1_CDE_11242019.pdf',bbox_inches='tight')

def speed_angle_heterogeneity_panels(master_trajectory_dict, trajectory_dict_polar_interp, step_angle_coupling_dict):
	
	#This is a wrapper for panels C-E in Figure 2
	
	fig = pt.figure(figsize = (7.5,4))
	
	####Specify [left,bottom,width,height] for each sub-panel
	
	pad = .26
	
	ncol = 3
	nrows = 2
	
	wstep = 1/ncol
	hstep = 1/nrows
	
	p_width = 1/ncol - 1/ncol*pad
	p_height = 1/nrows - 1/nrows*0
	
	
	#rect_locations = [[0,2*hstep,p_width,p_height],[.48*wstep,2*hstep+.2*hstep,p_width/3,p_height/3],[wstep,2*hstep,p_width,p_height],[2*wstep,2*hstep,p_width,p_height]]
	rect_locations = []
	###next 4 rows
	
	pad2 = .1
	
	w1 = 3/5.
	#w1 = 1/2.
	w2 = 1/5.
	
	#h2 = (2/3 - .1*2/3)*1/6
	h2 = 1/5
	p_width2 = w1 - w2*pad
	p_width3 = w2 - w2*pad
	
	p_height2 = h2 - h2*pad
	
	for i in range(4):
		rect_locations.append( [0,(3-i)*h2,p_width2,p_height2] )
		rect_locations.append( [w1,(3-i)*h2,p_width3,p_height2] )
		rect_locations.append( [w1+w2,(3-i)*h2,p_width3,p_height2] )
			
	####Create the requisite number of axis objects
	
	my_axes = []
	for rect_loc in rect_locations:
		
		ax = fig.add_axes(rect_loc)
		my_axes.append(ax)
	
	figure_panel_functions_11242019.speed_angle_heterogeneity_panels_revised(speeds_dict, trajectory_dict_polar, step_angle_coupling_dict, axis_lists=[[my_axes[0],my_axes[1],my_axes[2]],[my_axes[3],my_axes[4],my_axes[5]],[my_axes[6],my_axes[7],my_axes[8]],[my_axes[9],my_axes[10],my_axes[11]]])
	print('Done with speed heterogeneity')
	
	pt.tight_layout()	

	#pt.savefig('/mnt/c/Users/ejerison/Dropbox/imaging_data/figures/paper_figures/Figure2_CDE_11242019.pdf',bbox_inches='tight')
	
def speed_decorrelation(trajectory_dict_polar, speeds_dict):
	fig = pt.figure(figsize = (4,3))
	
	ax = pt.gca()
	figure_panel_functions_11242019.speed_rank_corr(trajectory_dict_polar, speeds_dict, ax)
	print('finished with speed decorrelation')
	#pt.savefig('/mnt/c/Users/ejerison/Dropbox/imaging_data/figures/paper_figures/Figure2_B_11242019.pdf',bbox_inches='tight')
	

def speed_class_and_manifold(master_trajectory_dict, speeds_dict, step_angle_coupling_dict, trajectory_dict_polar_interp):
	
	fig = pt.figure(figsize = (8,8))
	
	####Specify [left,bottom,width,height] for each sub-panel
	
	pad = .2
	
	ncol = 2
	nrows = 2
	
	wstep = 1/ncol
	hstep = 1/nrows
	
	p_width = 1/ncol - 1/ncol*pad
	p_height = 1/nrows - 1/nrows*pad
	
	rect_locations = [[0,hstep,p_width,p_height],[wstep,hstep,p_width,p_height],[0,0,p_width,p_height],
	                  [p_width+.05,0,p_width,p_height]]
	
	####Create the requisite number of axis objects
	
	my_axes = []
	ax_counter = 0
	axes_to_add = [0,1,2,3]
	for ind in axes_to_add:
		rect_loc = rect_locations[ind]
		if ind == 3:
			ax = fig.add_axes(rect_loc,projection='3d')
		else:
			ax = fig.add_axes(rect_loc)
		my_axes.append(ax)
		ax_counter += 1
	
	###Get speed classes
	
	speed_class_inds = figure_panel_functions_11242019.define_speed_classes( speeds_dict, step_angle_coupling_dict, master_trajectory_dict)
	
	####Make each of the panel plots
	
	figure_panel_functions_11242019.turn_angles_by_speed_class( step_angle_coupling_dict, speeds_dict,speed_class_inds, ax = my_axes[0])
	print('finished with turn angle dist')
	figure_panel_functions_11242019.speed_angle_coupling_by_speed_class( step_angle_coupling_dict, speeds_dict, 45, speed_class_inds, ax = my_axes[1])
	print('finished with speed-angle coupling')
	figure_panel_functions_11242019.MSDs_by_speed_class_fishonly(speeds_dict, step_angle_coupling_dict, master_trajectory_dict, speed_class_inds, ax = my_axes[2] )
	print('finished with MSDs')
	figure_panel_functions_11242019.all_cells_manifold_3d( step_angle_coupling_dict, speeds_dict, ax=my_axes[3] )
	print('finished 3D manifold')
	
	panel_letters = ['A','B','C','D']
	
	for i in range(len(my_axes)):
		if i != 3:
			ax = my_axes[i]
			ax.text( -.1,1.06, panel_letters[i], fontsize=14, transform=ax.transAxes, fontname="Arial")
			ax.tick_params(axis='both', which='major', labelsize=8)
			ax.tick_params(axis='both', which='minor', labelsize=8)
		else:
			ax = my_axes[i]
			ax.text2D( 0,1.06, panel_letters[i], fontsize=14, transform=ax.transAxes, fontname="Arial")
			ax.tick_params(axis='both', which='major', labelsize=8)
			ax.tick_params(axis='both', which='minor', labelsize=8)
	#figure_panel_functions_11242019.PSDs_by_speed_class(
	
	####Give each panel plot a label
	
	
	#pt.savefig('/mnt/c/Users/ejerison/Dropbox/imaging_data/figures/paper_figures/Figure3_ABCD_11242019.pdf',bbox_inches='tight')

def speed_class_collapse_trial(master_trajectory_dict, speeds_dict, step_angle_coupling_dict, trajectory_dict_polar_interp):
	
	fig = pt.figure(figsize = (8,8))
	
	####Specify [left,bottom,width,height] for each sub-panel
	
	pad = .2
	
	ncol = 2
	nrows = 2
	
	wstep = 1/ncol
	hstep = 1/nrows
	
	p_width = 1/ncol - 1/ncol*pad
	p_height = 1/nrows - 1/nrows*pad
	
	rect_locations = [[0,hstep,p_width,p_height],[wstep,hstep,p_width,p_height],[0,0,p_width,p_height],
	                  [p_width+.05,0,p_width,p_height]]
	
	####Create the requisite number of axis objects
	
	my_axes = []
	ax_counter = 0
	axes_to_add = [0,1,2,3]
	for ind in axes_to_add:
		rect_loc = rect_locations[ind]
		if ind == 3:
			ax = fig.add_axes(rect_loc,projection='3d')
		else:
			ax = fig.add_axes(rect_loc)
		my_axes.append(ax)
		ax_counter += 1
	
	###Get speed classes
	
	speed_class_inds = figure_panel_functions_11242019.define_speed_classes( speeds_dict, step_angle_coupling_dict, master_trajectory_dict)
	
	####Make each of the panel plots
	
	
	figure_panel_functions_11242019.MSDs_by_speed_class_fishonly_scaled(speeds_dict, step_angle_coupling_dict, master_trajectory_dict, speed_class_inds, ax = my_axes[2] )
	
	panel_letters = ['A','B','C','D']
	
	for i in range(len(my_axes)):
		if i != 3:
			ax = my_axes[i]
			ax.text( -.1,1.06, panel_letters[i], fontsize=14, transform=ax.transAxes, fontname="Arial")
			ax.tick_params(axis='both', which='major', labelsize=8)
			ax.tick_params(axis='both', which='minor', labelsize=8)
		else:
			ax = my_axes[i]
			ax.text2D( 0,1.06, panel_letters[i], fontsize=14, transform=ax.transAxes, fontname="Arial")
			ax.tick_params(axis='both', which='major', labelsize=8)
			ax.tick_params(axis='both', which='minor', labelsize=8)
	#figure_panel_functions_11242019.PSDs_by_speed_class(
	
	####Give each panel plot a label
	
	
	#pt.savefig('/mnt/c/Users/ejerison/Dropbox/imaging_data/figures/paper_figures/MSD_collapse_trial_12012019.pdf',bbox_inches='tight')
	
def persistence_time_and_MSDvspeed(master_trajectory_dict, speeds_dict, step_angle_coupling_dict, trajectory_dict_polar_interp):
	
	fig = pt.figure(figsize = (8,3))
	
	####Specify [left,bottom,width,height] for each sub-panel
	
	pad = .2
	
	ncol = 2
	nrows = 1
	
	wstep = 1/ncol
	hstep = 1/nrows
	
	p_width = 1/ncol - 1/ncol*pad
	p_height = 1/nrows #- 1/nrows*pad
	
	rect_locations = [[0,0,p_width,p_height],
					  [wstep,0,p_width,.7*p_height],[wstep,.72*p_height,p_width,.28*p_height]]
	
	####Create the requisite number of axis objects
	
	my_axes = []
	ax_counter = 0
	axes_to_add = [0,1,2]
	for ind in axes_to_add:
		rect_loc = rect_locations[ind]
		if ind == 4:
			ax = fig.add_axes(rect_loc,projection='3d')
		else:
			ax = fig.add_axes(rect_loc)
		my_axes.append(ax)
		ax_counter += 1
	
	figure_panel_functions_11242019.persistence_time_binned_fishonly( corr_times_dict, speeds_dict, ax = my_axes[0] )
	print('finished with persistence time')
	figure_panel_functions_11242019.MSD_v_speed(speeds_dict, corr_times_dict, master_trajectory_dict, ax = my_axes[1])
	print('finished with MSDs vs speed1')
	figure_panel_functions_11242019.MSD_v_speed_deviations(speeds_dict, corr_times_dict, master_trajectory_dict, ax = my_axes[2])
	print('finished with MSDs vs speed2')
	
	panel_letters = ['A','dummy','B','D','F','F','G','H','I','J','K','L']
	
	for i in range(len(my_axes)):
		if i == 0:
			ax = my_axes[i]
			ax.text( -.1,1.06, panel_letters[i], fontsize=14, transform=ax.transAxes, fontname="Arial")
			ax.tick_params(axis='both', which='major', labelsize=8)
			ax.tick_params(axis='both', which='minor', labelsize=8)
		elif i== 2:
			ax = my_axes[i]
			ax.text( -.1,1+.06*1/.28, panel_letters[i], fontsize=14, transform=ax.transAxes, fontname="Arial")
			ax.tick_params(axis='both', which='major', labelsize=8)
			ax.tick_params(axis='both', which='minor', labelsize=8)
	#figure_panel_functions_11242019.PSDs_by_speed_class(
	
	####Give each panel plot a label
	
	
	#pt.savefig('/mnt/c/Users/ejerison/Dropbox/imaging_data/figures/paper_figures/Figure4_11242019.pdf',bbox_inches='tight')
	
def plot_figure2(master_trajectory_dict, speeds_dict, step_angle_coupling_dict, trajectory_dict_polar_interp):
	
	fig = pt.figure(figsize = (10,10))
	
	####Specify [left,bottom,width,height] for each sub-panel
	
	pad = .2
	
	ncol = 3
	nrows = 3
	
	wstep = 1/ncol
	hstep = 1/nrows
	
	p_width = 1/ncol - 1/ncol*pad
	p_height = 1/nrows - 1/nrows*pad
	
	rect_locations = [[0,2*hstep,p_width,p_height],[wstep,2*hstep,p_width,p_height],[2*wstep,2*hstep,p_width,p_height],[0,hstep,p_width,p_height],
	                  [wstep,hstep,p_width,p_height],[2*wstep,hstep,p_width,.7*p_height],[2*wstep,hstep+.75*p_height,p_width,.25*p_height],
					  [0,0,p_width,.7*p_height],[.1*wstep,.1*hstep,p_width/3,p_height/3]]
	
	####Create the requisite number of axis objects
	
	my_axes = []
	ax_counter = 0
	axes_to_add = [0,1,2,3,5,6]
	for ind in axes_to_add:
		rect_loc = rect_locations[ind]
		if ind == 4:
			ax = fig.add_axes(rect_loc,projection='3d')
		else:
			ax = fig.add_axes(rect_loc)
		my_axes.append(ax)
		ax_counter += 1
	
	###Get speed classes
	
	speed_class_inds = figure_panel_functions_11242019.define_speed_classes( speeds_dict, step_angle_coupling_dict, master_trajectory_dict)
	
	####Make each of the panel plots
	
	figure_panel_functions_11242019.turn_angles_by_speed_class( step_angle_coupling_dict, speeds_dict,speed_class_inds, ax = my_axes[0])
	print('finished with turn angle dist')
	figure_panel_functions_11242019.speed_angle_coupling_by_speed_class( step_angle_coupling_dict, speeds_dict, 45, speed_class_inds, ax = my_axes[1])
	print('finished with speed-angle coupling')
	figure_panel_functions_11242019.persistence_time_binned_fishonly( corr_times_dict, speeds_dict, ax = my_axes[2] )
	print('finished with persistence time')
	figure_panel_functions_11242019.MSDs_by_speed_class_fishonly(speeds_dict, step_angle_coupling_dict, master_trajectory_dict, speed_class_inds, ax = my_axes[3] )
	print('finished with MSDs')
	figure_panel_functions_11242019.MSD_v_speed(speeds_dict, corr_times_dict, master_trajectory_dict, ax = my_axes[4])
	print('finished with MSDs vs speed1')
	figure_panel_functions_11242019.MSD_v_speed_deviations(speeds_dict, corr_times_dict, master_trajectory_dict, ax = my_axes[5])
	print('finished with MSDs vs speed2')
	
	panel_letters = ['A','B','C','D','F','F','G','H','I','J','K','L']
	
	for i in range(len(my_axes)):
		if i != 4:
			ax = my_axes[i]
			ax.text( -.1,1.06, panel_letters[i], fontsize=14, transform=ax.transAxes, fontname="Arial")
			ax.tick_params(axis='both', which='major', labelsize=8)
			ax.tick_params(axis='both', which='minor', labelsize=8)
		
	
	#pt.savefig('/mnt/c/Users/ejerison/Dropbox/imaging_data/figures/paper_figures/Figure2_noE_09202019.pdf',bbox_inches='tight')

def plot_figure2_subsampled(master_trajectory_dict, speeds_dict, step_angle_coupling_dict, trajectory_dict_polar_interp):
	
	fig = pt.figure(figsize = (10,10))
	
	####Specify [left,bottom,width,height] for each sub-panel
	
	pad = .2
	
	ncol = 3
	nrows = 3
	
	wstep = 1/ncol
	hstep = 1/nrows
	
	p_width = 1/ncol - 1/ncol*pad
	p_height = 1/nrows - 1/nrows*pad
	
	rect_locations = [[0,2*hstep,p_width,p_height],[wstep,2*hstep,p_width,p_height],[2*wstep,2*hstep,p_width,p_height],[0,hstep,p_width,p_height],
	                  [wstep,hstep,p_width,p_height],[2*wstep,hstep,p_width,.7*p_height],[2*wstep,hstep+.75*p_height,p_width,.25*p_height],
					  [0,0,p_width,.7*p_height],[.1*wstep,.1*hstep,p_width/3,p_height/3]]
	
	####Create the requisite number of axis objects
	
	my_axes = []
	ax_counter = 0
	axes_to_add = [0,1,2,3,5,6]
	for ind in axes_to_add:
		rect_loc = rect_locations[ind]
		if ind == 4:
			ax = fig.add_axes(rect_loc,projection='3d')
		else:
			ax = fig.add_axes(rect_loc)
		my_axes.append(ax)
		ax_counter += 1
	
	###Get speed classes
	
	speed_class_inds = figure_panel_functions_11242019.define_speed_classes_subsamp( speeds_dict, step_angle_coupling_dict, master_trajectory_dict)
	
	####Make each of the panel plots
	
	figure_panel_functions_11242019.turn_angles_by_speed_class( step_angle_coupling_dict, speeds_dict,speed_class_inds, ax = my_axes[0])
	print('finished with turn angle dist')
	figure_panel_functions_11242019.speed_angle_coupling_by_speed_class( step_angle_coupling_dict, speeds_dict, 90, speed_class_inds, ax = my_axes[1])
	print('finished with speed-angle coupling')
	figure_panel_functions_11242019.persistence_time_binned_fishonly( corr_times_dict, speeds_dict, ax = my_axes[2] )
	print('finished with persistence time')
	figure_panel_functions_11242019.MSDs_by_speed_class_fishonly_sub(speeds_dict, step_angle_coupling_dict, master_trajectory_dict, speed_class_inds, ax = my_axes[3] )
	print('finished with MSDs')
	figure_panel_functions_11242019.MSD_v_speed(speeds_dict, corr_times_dict, master_trajectory_dict, ax = my_axes[4])
	print('finished with MSDs vs speed1')
	figure_panel_functions_11242019.MSD_v_speed_deviations(speeds_dict, corr_times_dict, master_trajectory_dict, ax = my_axes[5])
	print('finished with MSDs vs speed2')
	
	panel_letters = ['A','B','C','D','F','F','G','H','I','J','K','L']
	
	for i in range(len(my_axes)):
		if i != 4:
			ax = my_axes[i]
			ax.text( -.1,1.06, panel_letters[i], fontsize=14, transform=ax.transAxes, fontname="Arial")
			ax.tick_params(axis='both', which='major', labelsize=8)
			ax.tick_params(axis='both', which='minor', labelsize=8)
		
	#figure_panel_functions_11242019.PSDs_by_speed_class(
	
	####Give each panel plot a label
	
	
	#pt.savefig('/mnt/c/Users/ejerison/Dropbox/imaging_data/figures/paper_figures/Figure2_09202019.pdf',bbox_inches='tight')
	

def plot_figure1(master_trajectory_dict, trajectory_dict_polar_interp, step_angle_coupling_dict):
	
	fig = pt.figure(figsize = (8,6))
	
	####Specify [left,bottom,width,height] for each sub-panel
	
	pad = .26
	
	ncol = 3
	nrows = 3
	
	wstep = 1/ncol
	hstep = 1/nrows
	
	p_width = 1/ncol - 1/ncol*pad
	p_height = 1/nrows - 1/nrows*0
	
	###first row
	rect_locations = [[0,2*hstep,p_width,p_height],[.48*wstep,2*hstep+.2*hstep,p_width/3,p_height/3],[wstep,2*hstep,p_width,p_height],[2*wstep,2*hstep,p_width,p_height]]
	
	###next 4 rows
	
	pad2 = .1
	
	w1 = 3/5.
	#w1 = 1/2.
	w2 = 1/5.
	
	h2 = (2/3 - .1*2/3)*1/6
	
	p_width2 = w1 - w2*pad
	p_width3 = w2 - w2*pad
	
	p_height2 = h2 - h2*pad
	
	for i in range(4):
		rect_locations.append( [0,(3-i)*h2,p_width2,p_height2] )
		rect_locations.append( [w1,(3-i)*h2,p_width3,p_height2] )
		rect_locations.append( [w1+w2,(3-i)*h2,p_width3,p_height2] )
			
	####Create the requisite number of axis objects
	
	my_axes = []
	for rect_loc in rect_locations:
		
		ax = fig.add_axes(rect_loc)
		my_axes.append(ax)
	#print('Starting MSD calculation')
	#figure_panel_functions_11242019.MSD_overall(speeds_dict, master_trajectory_dict, ax=my_axes[0],inset_ax=my_axes[1])
	#print('Done calculating overall MSDs')
	#figure_panel_functions_11242019.PSDs_overall(corr_times_dict, trajectory_dict_polar_interp, ax=my_axes[2])
	#print('Done calculating overall PSDs')
	#figure_panel_functions_11242019.boutx_distribution( speeds_dict,trajectory_dict_polar_interp,ax=my_axes[3])
	#print('Done calculating boutx')
	figure_panel_functions_11242019.speed_angle_heterogeneity_panels_revised(speeds_dict, trajectory_dict_polar, step_angle_coupling_dict, axis_lists=[[my_axes[4],my_axes[5],my_axes[6]],[my_axes[7],my_axes[8],my_axes[9]],[my_axes[10],my_axes[11],my_axes[12]],[my_axes[13],my_axes[14],my_axes[15]]])
	print('Done with speed heterogeneity')
	
	panel_letters = ['C','dummy','D','dummy','E']
	
	#for i in [0,2,4]:
		
		#ax = my_axes[i]
		
		#ax.text( -.1,1.06, panel_letters[i], fontsize=14, transform=ax.transAxes, fontname="Arial")
		#ax.tick_params(axis='both', which='major', labelsize=8)
		#ax.tick_params(axis='both', which='minor', labelsize=8)
	#pt.tight_layout()	
	
	#pt.savefig('/mnt/c/Users/ejerison/Dropbox/imaging_data/figures/paper_figures/Figure2_BCD_11052019.pdf',bbox_inches='tight')


def plot_figure4_mouse_dicty(speeds_dict, corr_times_dict):
	
	fig = pt.figure(figsize = (7,6))
	
	####Specify [left,bottom,width,height] for each sub-panel
	
	pad = .2
	
	ncol = 2
	nrows = 2
	
	wstep = 1/ncol
	hstep = 1/nrows
	
	p_width = 1/ncol - 1/ncol*pad
	p_height = 1/nrows - 1/nrows*pad
	
	rect_locations = [[0,hstep,p_width,p_height],[wstep,hstep,p_width,p_height],[0,0,p_width,p_height],[wstep,0,p_width,p_height]]#0,2*hstep,p_width,p_height],[wstep,2*hstep,p_width,p_height],
	
	####Create the requisite number of axis objects
	
	my_axes = []
	for rect_loc in rect_locations:
		
		ax = fig.add_axes(rect_loc)
		my_axes.append(ax)
	
	####Make each of the panel plots
	
	
	figure_panel_functions_11242019.angle_speed_correlations_with_perturbations_all_cells( speeds_dict, step_angle_coupling_dict, experiment='Gerard_mouse_T', ax = my_axes[0], ax2 = my_axes[2])
	figure_panel_functions_11242019.angle_speed_correlations_with_perturbations_all_cells( speeds_dict, step_angle_coupling_dict, experiment='Gautreau_dicty', ax = my_axes[1], ax2 = my_axes[3])
	
	
	panel_letters = ['A','B','C','D','E','F','G','H','I']
	
	for i in range(len(my_axes)):
		
		ax = my_axes[i]
		ax.text( -.1,1.06, panel_letters[i], fontsize=14, transform=ax.transAxes, fontname="Arial")
		ax.tick_params(axis='both', which='major', labelsize=8)
		ax.tick_params(axis='both', which='minor', labelsize=8)
	
	####Give each panel plot a label
	my_axes[0].set_title('M. musculus',style='italic')
	my_axes[1].set_title('Dictyostelium',style='italic')
	#pt.savefig('/mnt/c/Users/ejerison/Dropbox/imaging_data/figures/paper_figures/Figure4_mousedicty_09212019.pdf',bbox_inches='tight')

def plot_figure3_fish(speeds_dict, corr_times_dict):
	
	fig = pt.figure(figsize = (4,7))
	
	####Specify [left,bottom,width,height] for each sub-panel
	
	pad = .2
	
	ncol = 1
	nrows = 2
	
	wstep = 1/ncol
	hstep = 1/nrows
	
	p_width = 1/ncol - 1/ncol*pad
	p_height = 1/nrows - 1/nrows*pad
	
	rect_locations = [[0,hstep,p_width,p_height],[0,0,p_width,p_height]]#[0,2*hstep,p_width,p_height],
	
	####Create the requisite number of axis objects
	
	my_axes = []
	for rect_loc in rect_locations:
		
		ax = fig.add_axes(rect_loc)
		my_axes.append(ax)
	
	####Make each of the panel plots
	
	
	figure_panel_functions_11242019.angle_speed_correlations_with_perturbations_all_cells( speeds_dict, step_angle_coupling_dict, experiment='fish_T', ax = my_axes[0], ax2 = my_axes[1])
	
	panel_letters = ['C','D','E','F','G','H','I']
	
	#for i in range(len(my_axes)):
		
	#	ax = my_axes[i]
	#	ax.text( -.1,1.06, panel_letters[i], fontsize=14, transform=ax.transAxes, fontname="Arial")
	#	ax.tick_params(axis='both', which='major', labelsize=8)
	#	ax.tick_params(axis='both', which='minor', labelsize=8)
	
	####Give each panel plot a label
	
	#pt.savefig('/mnt/c/Users/ejerison/Dropbox/imaging_data/figures/paper_figures/Figure3_fish_rockout_09202019.pdf',bbox_inches='tight')


def plot_manifold_3D(speeds_dict, step_angle_coupling_dict):
	###This is figure 2E, but the spacing comes out funny if it is included as a panel in figure 2
	fig = pt.figure(figsize=(5,5))
	my_ax = fig.add_subplot(111, projection='3d')
	figure_panel_functions_11242019.all_cells_manifold_3d( step_angle_coupling_dict, speeds_dict, ax=my_ax )
	#pt.show()
	#pt.savefig('/mnt/c/Users/ejerison/Dropbox/imaging_data/figures/paper_figures/Figure3D_manifold_09202019.pdf')

def plot_manifold_3D_subsamp(speeds_dict, step_angle_coupling_dict):
	fig = pt.figure(figsize=(5,5))
	my_ax = fig.add_subplot(111, projection='3d')
	figure_panel_functions_11242019.all_cells_manifold_3d( step_angle_coupling_dict, speeds_dict, ax=my_ax )
	#pt.show()
	#pt.savefig('/mnt/c/Users/ejerison/Dropbox/imaging_data/figures/paper_figures/Figure3D_manifold_subsamp_09202019.pdf')
	

def plot_anova(step_angle_coupling_dict, speeds_dict):
	fig, (ax1,ax2) = pt.subplots(1,2,figsize=(9,4))
	figure_panel_functions_11242019.manifold_anova(step_angle_coupling_dict, speeds_dict, ax1, ax2 )
	panel_letters = ['A','B','C','D','E','F','G','H','I']
	my_axes = [ax1,ax2]
	for i in range(len(my_axes)):
		
		ax = my_axes[i]
		ax.text( -.1,1.06, panel_letters[i], fontsize=14, transform=ax.transAxes, fontname="Arial")
		
	#pt.savefig('/mnt/c/Users/ejerison/Dropbox/imaging_data/figures/paper_figures/speed_angle_spline_for_anova_09202019.pdf',bbox_inches='tight')
	
def plot_figure3_pairedonly(speeds_dict, corr_times_dict):
	
	fig = pt.figure(figsize = (4,7))
	
	####Specify [left,bottom,width,height] for each sub-panel
	
	pad = .2
	
	ncol = 1
	nrows = 2
	
	wstep = 1/ncol
	hstep = 1/nrows
	
	p_width = 1/ncol - 1/ncol*pad
	p_height = 1/nrows - 1/nrows*pad
	
	rect_locations = [[0,hstep,p_width,p_height],[0,0,p_width,p_height]]#[0,2*hstep,p_width,p_height],
	
	####Create the requisite number of axis objects
	
	my_axes = []
	for rect_loc in rect_locations:
		
		ax = fig.add_axes(rect_loc)
		my_axes.append(ax)
	
	####Make each of the panel plots
	
	
	figure_panel_functions_11242019.angle_speed_correlations_with_perturbations_paired_samples( speeds_dict, step_angle_coupling_dict, experiment='fish_T', ax = my_axes[0], ax2 = my_axes[1])
	
	panel_letters = ['C','D','E','F','G','H','I']
	
	#for i in range(len(my_axes)):
		
	#	ax = my_axes[i]
	#	ax.text( -.1,1.06, panel_letters[i], fontsize=14, transform=ax.transAxes, fontname="Arial")
	#	ax.tick_params(axis='both', which='major', labelsize=8)
	#	ax.tick_params(axis='both', which='minor', labelsize=8)
	
	####Give each panel plot a label
	
	#pt.savefig('/mnt/c/Users/ejerison/Dropbox/imaging_data/figures/paper_figures/Figure3_fish_pairedonly_09202019.pdf',bbox_inches='tight')

def plot_psd_np(corr_times_dict, trajectory_dict_polar_interp, step_angle_coupling_dict):
	pt.figure()
	ax = pt.gca()
	figure_panel_functions_11242019.PSDs_overall_np(corr_times_dict, trajectory_dict_polar_interp, ax=ax)
	pt.savefig('/mnt/c/Users/ejerison/Dropbox/imaging_data/figures/psd_np_check.pdf')

def plot_msd_overall_shorter(speeds_dict, master_trajectory_dict):
	fig = pt.figure(figsize=(4,3))
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
	ax = pt.gca()
	figure_panel_functions_11242019.MSD_overall_shorter(speeds_dict, master_trajectory_dict, ax=my_axes[0],inset_ax=my_axes[1])
	pt.savefig('/Users/ejerison/Dropbox/imaging_data/figures/paper_figures/Figure1_Supp1_MSDshorter.pdf',bbox_inches='tight')
	
###Call the appropriate figure functions to plot a figure!

#Figure 1
#MSD_PSD_boutx_pooled_panels(master_trajectory_dict, trajectory_dict_polar_interp, step_angle_coupling_dict)
#Figure 2
speed_angle_heterogeneity_panels(master_trajectory_dict, trajectory_dict_polar_interp, step_angle_coupling_dict)
#speed_decorrelation(trajectory_dict_polar, speeds_dict)
#Figure 3
#speed_class_and_manifold(master_trajectory_dict, speeds_dict, step_angle_coupling_dict, trajectory_dict_polar_interp)
#Figure 4
#persistence_time_and_MSDvspeed(master_trajectory_dict, speeds_dict, step_angle_coupling_dict, trajectory_dict_polar_interp)

#plot_msd_overall_shorter(speeds_dict, master_trajectory_dict)
#plot_figure2_subsampled(master_trajectory_dict_subsamp,speeds_dict_subsamp, step_angle_coupling_dict_subsamp,trajectory_dict_polar_interp_subsamp)
#plot_manifold_3D(speeds_dict, step_angle_coupling_dict)
#plot_manifold_3D_subsamp(speeds_dict_subsamp, step_angle_coupling_dict_subsamp)
#plot_anova(step_angle_coupling_dict, speeds_dict)

#plot_figure3_fish(speeds_dict, corr_times_dict)
#plot_figure3_pairedonly(speeds_dict, corr_times_dict)

#plot_figure4_mouse_dicty(speeds_dict, corr_times_dict)
#plot_psd_np(corr_times_dict, trajectory_dict_polar_interp, step_angle_coupling_dict)
#figure_panel_functions_11242019.speed_dist_angle_dist_KS_test( speeds_dict, step_angle_coupling_dict )