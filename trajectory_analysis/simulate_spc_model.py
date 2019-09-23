import numpy
from scipy import special
import scipy
from scipy.stats import binned_statistic
import scipy.special

def calc_dv(v,v0,F,D):
	indicator1 = numpy.tile(numpy.array(numpy.sqrt(numpy.sum(v**2,axis=1)) < v0,dtype='int'),(2,1)).T
	indicator2 = numpy.tile(numpy.array(numpy.sqrt(numpy.sum(v**2,axis=1)) > v0,dtype='int'),(2,1)).T
	#indicator1 = numpy.array(numpy.abs(v) < v0,dtype='int')
	#indicator2 = numpy.array(numpy.abs(v) > v0,dtype='int')
	dv = -F/v0*(v*indicator1 + v0*numpy.sign(v)*indicator2) + numpy.random.normal(loc=0,scale=numpy.sqrt(D),size=v.shape)
	
	return dv

def calc_dv_logistic(v,v0,F,D):
	
	###expit: 1/(1+exp(-x))
	
	dv = -F*(2*(scipy.special.expit(v/v0)-.5)) + numpy.random.normal(loc=0,scale=numpy.sqrt(D),size=v.shape)
	
	return dv
	
def simulate_spc(ncells,nt,v0,F,D):
	
	v_init = numpy.zeros((ncells,2),dtype='float')
	#v_init = numpy.random.exponential(v0,size=(ncells,2))
	vt = [v_init]
	for t in range(nt):
		dv = calc_dv_logistic(vt[t],v0,F,D)
		#dv = calc_dv(vt[t],v0,F,D)
		v = vt[t] + dv
		vt.append(v)

	vt = numpy.array(vt)

	trajs = numpy.cumsum(vt, axis=0)
	
	return trajs, vt

def simulate_prw(ncells,nt,S,P,sigma_noise):
	
	v_init = numpy.zeros((ncells,2),dtype='float')
	vt = [v_init]
	for t in range(nt):
		
		dv = -1./P*vt[t] + numpy.random.normal(loc=0,scale=S/numpy.sqrt(P),size=vt[t].shape)
		v = vt[t] + dv
		vt.append(v)
	
	vt = numpy.array(vt)

	trajs = numpy.cumsum(vt, axis=0) + numpy.random.normal(loc=0,scale=sigma_noise,size=vt.shape)
	
	return trajs

def simulate_prw_scaled(ncells,nt,P,sigma_noise):

	###We simulate the prw with dimensionless v--i.e. v = v_real/S
	###The langevin equation is then: dv = -1/P*v*dt + 1/sqrt(P)*dW
	
	v_init = numpy.zeros((ncells,2),dtype='float')
	vt = [v_init]
	for t in range(nt):
		
		dv = -1./P*vt[t] + numpy.random.normal(loc=0,scale=1./numpy.sqrt(P),size=vt[t].shape)
		v = vt[t] + dv
		vt.append(v)
	
	vt = numpy.array(vt)

	trajs = numpy.cumsum(vt, axis=0) + numpy.random.normal(loc=0,scale=sigma_noise,size=vt.shape)
	
	return trajs

def simulate_rw_prw(ncells,nt,v0,p_sw1,p_sw2,S,P):
	
	v_init = numpy.zeros((ncells,2),dtype='float')
	vt = [v_init]
	cell_indicators = numpy.ones((ncells,2),dtype='int')
	#sigma = S/P*2
	sigma = S/numpy.sqrt(P)
	for t in range(nt):
	
		###Update velocities for this timestep; for non-persistent 'pause' periods, velocities are normal random variables with std dev S/sqrt(P)
		dv_persistent = -1./P*vt[t] + numpy.random.normal(loc=0,scale=S/numpy.sqrt(P),size=vt[t].shape)
		#v_notpersistent = numpy.random.normal(loc=0,scale=S/numpy.sqrt(P),size=vt[t].shape)
		v_notpersistent = numpy.random.normal(loc=0,scale=sigma,size=vt[t].shape)
		
		v = vt[t]*cell_indicators + dv_persistent*cell_indicators + v_notpersistent*(1-cell_indicators)
		vt.append(v)
		###Update whether or not cells are in the persistent or non-persistent mode
		
		switches1 = numpy.random.poisson(p_sw1,size=ncells)
		switches2 = numpy.random.poisson(p_sw2,size=ncells)
		cell_indicators_new = numpy.abs( cell_indicators - cell_indicators*numpy.tile(numpy.array(numpy.logical_and(switches1 > .5,numpy.sqrt(numpy.sum(v**2,axis=1)) < v0),dtype='int'),(2,1)).T + (1-cell_indicators)*numpy.tile(numpy.array(numpy.logical_and(switches2 > .5,numpy.sqrt(numpy.sum(v**2,axis=1)) < v0),dtype='int'),(2,1)).T)
		cell_indicators = cell_indicators_new
	
	vt = numpy.array(vt)

	trajs = numpy.cumsum(vt, axis=0)
	
	return trajs, vt
	
def simulate_levy(ncells,nt,v0,gamma,lambda0,lambdaf):
	###This is a simple implementation of a Levy-style random walk. Every 100 timesteps, cells choose steps from a heavy-tailed distribution.
	###Observations are modeled as a discretization of these trajectories.
	###Because the heavy-tailed distribution is not normalizable, for simplicity we will simply set it to a constant value on 0-lambda0 and cut it off at lambdaf.
	
	###Construct the sampling distribution by inverting the CDF
	
	xrange_const = numpy.arange(.01,lambda0,.01)
	xrange_pl = numpy.arange(lambda0,lambdaf,.01)
	xvals_all = numpy.concatenate((xrange_const,xrange_pl))
	
	match_val = lambda0**(-1*gamma)
	pdf = numpy.concatenate( (match_val*numpy.ones((len(xrange_const,),),dtype='float'), xrange_pl**(-1*gamma)) )
	
	cdf = numpy.cumsum(pdf)/numpy.sum(pdf)
	xvals_all = numpy.insert(xvals_all,0,0)
	cdf = numpy.insert(cdf,0,0)
	inverse_cdf = scipy.interpolate.interp1d(cdf,xvals_all)
	
	###Record trajectories
	
	vlist = []
	nsteps = int(nt/100)
	
	for n in range(nsteps):
		thetas = 2*numpy.pi*numpy.random.uniform(0,1,size=ncells)
		disp_seeds = numpy.random.uniform(0,1,size=ncells)
		disps = v0*inverse_cdf(disp_seeds)
		
		xs = disps*numpy.cos(thetas)/100
		ys = disps*numpy.sin(thetas)/100
		
		for substep in range(100):
			vlist.append([xs,ys])
			
	v_arr = numpy.array(vlist)
	v_mat = numpy.swapaxes(v_arr,1,2)
	
	trajs = numpy.cumsum(v_mat,axis=0)
	
	return trajs,v_mat

def measure_disp_dist(polar_trajs,data_len,timescale):
	
	mean_speeds_sim = numpy.mean(polar_trajs[:,:,0],axis=1)#numpy.abs(polar_trajs[:,:,0]*numpy.cos(polar_trajs[:,:,1])),axis=0)
	ncellssub,ntsub,n=polar_trajs.shape
	disps = []
	local_speeds = []
	for n in range(ncellssub):
		for t in range(ntsub):
			disp = 0.
			theta0 = polar_trajs[n,t,1]
			j = t
			
			while j < ntsub and numpy.abs(polar_trajs[n,j,1] - theta0) < numpy.pi/2:
				disp += polar_trajs[n,j,0]*numpy.cos(polar_trajs[n,j,1])*numpy.cos(theta0) + polar_trajs[n,j,0]*numpy.sin(polar_trajs[n,j,1])*numpy.sin(theta0)
				j += 1
			if j < ntsub:
				disps.append(disp/(mean_speeds_sim[n]*timescale))
				local_speeds.append(polar_trajs[n,t,0]/mean_speeds_sim[n])
				
	disp_bins = numpy.percentile(disps,numpy.arange(0,101,int(100/data_len)))
	disp_dist,nbins = numpy.histogram(disps,bins=disp_bins,density=True)
	xlocs,binedges,nbins = binned_statistic(disps,disps,bins=disp_bins)
	
	return disp_dist, xlocs

def measure_persistence_times(polar_trajs,timescale):
	
	mean_speeds_sim = numpy.mean(polar_trajs[:,:,0],axis=1)#numpy.abs(polar_trajs[:,:,0]*numpy.cos(polar_trajs[:,:,1])),axis=0)
	ncellssub,ntsub,n=polar_trajs.shape
	disps = []
	local_speeds = []
	for n in range(ncellssub):
		for t in range(ntsub):
			disp = 0.
			theta0 = polar_trajs[n,t,1]
			j = t+1
			
			while j < ntsub and numpy.abs(polar_trajs[n,j,1] - theta0) < numpy.pi/2:
				disp += 1#polar_trajs[n,j,0]*numpy.cos(polar_trajs[n,j,1])*numpy.cos(theta0) + polar_trajs[n,j,0]*numpy.sin(polar_trajs[n,j,1])*numpy.sin(theta0)
				j += 1
			if j < ntsub:
				disps.append(disp)
				local_speeds.append(polar_trajs[n,t,0]/mean_speeds_sim[n])
	
	return disps

def measure_persistence_times_by_traj(polar_trajs,timescale):
	
	mean_speeds_sim = numpy.mean(polar_trajs[:,:,0],axis=1)#numpy.abs(polar_trajs[:,:,0]*numpy.cos(polar_trajs[:,:,1])),axis=0)
	ncellssub,ntsub,n=polar_trajs.shape
	disps = []
	local_speeds = []
	for n in range(ncellssub):
		disps_cell = []
		for t in range(ntsub):
			disp = 0.
			theta0 = polar_trajs[n,t,1]
			j = t+1
			
			while j < ntsub and numpy.abs(polar_trajs[n,j,1] - theta0) < numpy.pi/2:
				disp += 1#polar_trajs[n,j,0]*numpy.cos(polar_trajs[n,j,1])*numpy.cos(theta0) + polar_trajs[n,j,0]*numpy.sin(polar_trajs[n,j,1])*numpy.sin(theta0)
				j += 1
			if j < ntsub:
				disps_cell.append(disp)
				local_speeds.append(polar_trajs[n,t,0]/mean_speeds_sim[n])
		disps.append(disps_cell)
	return disps

def measure_disp_dist_scale2(polar_trajs,data_len,timescale):
	
	mean_speeds_sim = numpy.mean(polar_trajs[:,:,0],axis=1)#numpy.abs(polar_trajs[:,:,0]*numpy.cos(polar_trajs[:,:,1])),axis=0)
	ncellssub,ntsub,n=polar_trajs.shape
	disps = []
	local_speeds = []
	mean_speeds = []
	for n in range(ncellssub):
		for t in range(ntsub):
			disp = 0.
			theta0 = polar_trajs[n,t,1]
			j = t
			
			while j < ntsub and numpy.abs(polar_trajs[n,j,1] - theta0) < numpy.pi/2:
				disp += polar_trajs[n,j,0]*numpy.cos(polar_trajs[n,j,1])*numpy.cos(theta0) + polar_trajs[n,j,0]*numpy.sin(polar_trajs[n,j,1])*numpy.sin(theta0)
				j += 1
			if j < ntsub:
				disps.append(disp)
				local_speeds.append(polar_trajs[n,t,0]/mean_speeds_sim[n])
				mean_speeds.append(mean_speeds_sim[n])
	disp_stat = (numpy.array(disps) - numpy.mean(disps))/numpy.array(mean_speeds)**1.5
	disp_bins = numpy.percentile(disp_stat,numpy.arange(0,101,int(100/data_len)))
	disp_dist,nbins = numpy.histogram(disp_stat,bins=disp_bins,density=True)
	xlocs,binedges,nbins = binned_statistic(disp_stat,disp_stat,bins=disp_bins)
	
	return disp_dist, xlocs

def measure_disp_dist_scale3(polar_trajs,data_len,timescale):
	
	mean_speeds_sim = numpy.mean(polar_trajs[:,:,0],axis=1)#numpy.abs(polar_trajs[:,:,0]*numpy.cos(polar_trajs[:,:,1])),axis=0)
	ncellssub,ntsub,n=polar_trajs.shape
	disps = []
	local_speeds = []
	mean_speeds = []
	for n in range(ncellssub):
		for t in range(ntsub):
			disp = 0.
			theta0 = polar_trajs[n,t,1]
			j = t
			
			while j < ntsub and numpy.abs(polar_trajs[n,j,1] - theta0) < numpy.pi/2:
				disp += polar_trajs[n,j,0]*numpy.cos(polar_trajs[n,j,1])*numpy.cos(theta0) + polar_trajs[n,j,0]*numpy.sin(polar_trajs[n,j,1])*numpy.sin(theta0)
				j += 1
			if j < ntsub:
				disps.append(disp)
				local_speeds.append(polar_trajs[n,t,0]/mean_speeds_sim[n])
				mean_speeds.append(mean_speeds_sim[n])
	disp_stat = (numpy.array(disps) - numpy.mean(disps))/numpy.array(mean_speeds)**2
	disp_bins = numpy.percentile(disp_stat,numpy.arange(0,101,int(100/data_len)))
	disp_dist,nbins = numpy.histogram(disp_stat,bins=disp_bins,density=True)
	xlocs,binedges,nbins = binned_statistic(disp_stat,disp_stat,bins=disp_bins)
	
	return disp_dist, xlocs
	
def measure_coupling(polar_trajs,timescale):
	
	mean_speeds_sim = numpy.mean(polar_trajs[:,:,0],axis=1)#numpy.abs(polar_trajs[:,:,0]*numpy.cos(polar_trajs[:,:,1])),axis=0)
	ncellssub,ntsub,n=polar_trajs.shape
	disps = []
	local_speeds = []
	for n in range(ncellssub):
		for t in range(ntsub):
			disp = 0.
			theta0 = polar_trajs[n,t,1]
			j = t
			
			while j < ntsub and numpy.abs(polar_trajs[n,j,1] - theta0) < numpy.pi/2:
				disp += polar_trajs[n,j,0]*numpy.cos(polar_trajs[n,j,1])*numpy.cos(theta0) + polar_trajs[n,j,0]*numpy.sin(polar_trajs[n,j,1])*numpy.sin(theta0)
				j += 1
			if j < ntsub:
				disps.append(disp/(mean_speeds_sim[n]*timescale))
				local_speeds.append((polar_trajs[n,t,0]/mean_speeds_sim[n]*timescale))
	speed_disp_bins = numpy.percentile(local_speeds,numpy.arange(0,101,5))
	#speed_disp_bins = numpy.arange(.25,2.01,.25)
	binned_stat,bins,binedges = binned_statistic(local_speeds,disps,bins=speed_disp_bins)
	xlocs,binedges,nbins = binned_statistic(local_speeds,local_speeds,bins=speed_disp_bins)
	
	return binned_stat, xlocs

def measure_time_coupling(polar_trajs,timescale):
	
	mean_speeds_sim = numpy.mean(polar_trajs[:,:,0],axis=1)#numpy.abs(polar_trajs[:,:,0]*numpy.cos(polar_trajs[:,:,1])),axis=0)
	ncellssub,ntsub,n=polar_trajs.shape
	disps = []
	local_speeds = []
	for n in range(ncellssub):
		for t in range(ntsub):
			disp = 0.
			theta0 = polar_trajs[n,t,1]
			j = t
			
			while j < ntsub and numpy.abs(numpy.sign(numpy.cos(polar_trajs[n,j,1])) - numpy.sign(numpy.cos(polar_trajs[n,t,1]))) < .5:#numpy.abs(polar_trajs[n,j,1] - theta0) < numpy.pi/2:
				disp += 1#polar_trajs[n,j,0]*numpy.cos(polar_trajs[n,j,1])*numpy.cos(theta0) + polar_trajs[n,j,0]*numpy.sin(polar_trajs[n,j,1])*numpy.sin(theta0)
				j += 1
			if j < ntsub:
				disps.append(disp/timescale)#(mean_speeds_sim[n]*timescale))
				local_speeds.append((polar_trajs[n,t,0]/mean_speeds_sim[n]*timescale))
	speed_disp_bins = numpy.percentile(local_speeds,numpy.arange(0,101,5))
	#speed_disp_bins = numpy.arange(.25,2.01,.25)
	binned_stat,bins,binedges = binned_statistic(local_speeds,disps,bins=speed_disp_bins)
	xlocs,binedges,nbins = binned_statistic(local_speeds,local_speeds,bins=speed_disp_bins)
	
	return binned_stat, xlocs

		
def exGaussian(x,mu,sigma,exrate,norm):
	
	#f = exrate/2*numpy.exp(exrate/2*(2*mu + exrate*sigma**2 - 2*x))*special.erfc((mu+exrate*sigma**2 - x)/(sigma*numpy.sqrt(2)))
	f = norm*numpy.exp(exrate/2*(2*mu + exrate*sigma**2 - 2*x))*special.erfc((mu+exrate*sigma**2 - x)/(sigma*numpy.sqrt(2)))
	
	return f

def piecewise_exp_Gauss(x,x0,sigma,exrate,norm):
	
	f = norm*(numpy.heaviside(numpy.abs(x)-x0,numpy.ones_like(x))*numpy.exp(-1*exrate*numpy.abs(x)) + numpy.exp(-1*exrate*x0 + x0**2/(2*sigma**2))*numpy.heaviside(x0-numpy.abs(x),numpy.ones_like(x))*numpy.exp(-1*x**2/(2*sigma**2)))
	
	return numpy.log10(f)
	
def simulate_spc_plus_blebbing(ncells,nt,v0,F,D,sigmab):

	v_init = numpy.random.exponential(v0,size=(ncells,2))
	vt = [v_init]
	for t in range(nt):
		
		dv = calc_dv(vt[t],v0,F,D)
		vnorms = numpy.sqrt(numpy.sum(vt[t]**2,axis=1))
		thetas = numpy.arccos(vt[t][:,0]/vnorms)
		v = vt[t] + dv + numpy.random.normal(loc=-100*sigmab*(vt[t].T/vnorms).T, scale=sigmab)
		vt.append(v)

	vt = numpy.array(vt)

	trajs = numpy.cumsum(vt, axis=0)
	
	return trajs, vt

def calculate_polar_trajs(trajs,tau,tmeasure_range):
	###Calculate relative angles between displacement vectors over tau timesteps
	trajs_sub = trajs[tmeasure_range,:,:]
	disps = trajs_sub[tau:,:,:] - trajs_sub[:-1*tau,:,:]
	disps_ss = disps[numpy.arange(0,disps.shape[0],tau),:,:]
	lens = numpy.sqrt(numpy.sum(disps_ss**2,axis=2))
	thetas = numpy.sign(disps_ss[:,:,1])*numpy.arccos(disps_ss[:,:,0]/lens)
	a = numpy.swapaxes(numpy.array([lens,thetas]),0,2)
	return a
	
def calculate_relative_cosangles(trajs,tau,tmeasure_range):
	###Calculate relative angles between displacement vectors over tau timesteps
	trajs_sub = trajs[tmeasure_range,:,:]
	disps = trajs_sub[tau:,:,:] - trajs_sub[:-1*tau,:,:]
	disps_ss = disps[numpy.arange(0,disps.shape[0],tau),:,:]
	
	dot_prods = (disps_ss[1:,:,0]*disps_ss[:-1,:,0] + disps_ss[1:,:,1]*disps_ss[:-1,:,1])/numpy.sqrt( numpy.sum(disps_ss[1:,:,:]**2,axis=2)*numpy.sum(disps_ss[:-1,:,:]**2,axis=2) )
	
	return dot_prods, disps_ss

def calculate_msds_by_cell(trajs,tau,tmeasure_range):
	
	###Calculate msds 

	trajs_sub = trajs[tmeasure_range,:,:]
	disps = trajs_sub[tau:,:,:] - trajs_sub[:-1*tau,:,:]
	return numpy.mean( numpy.sum( disps**2, axis=2 ), axis=0 )

def calculate_msdx_dist(trajs,tau,tmeasure_range,data_len):
	
	###Calculate msds 

	trajs_sub = trajs[tmeasure_range,:,:]
	disps0 = trajs_sub[1:,:,:] - trajs_sub[:-1,:,:]
	disps = trajs_sub[tau:,:,:] - trajs_sub[:-1*tau,:,:]
	mean_speeds = numpy.mean(numpy.sqrt(numpy.sum(disps0**2,axis=2)),axis=0)
	print(mean_speeds.shape,disps.shape)
	rms = numpy.abs((disps[:,:,0]/mean_speeds).flatten())
	rms_bins = numpy.percentile(rms,numpy.arange(0,101,10))
	rms_dist,binedges = numpy.histogram(rms,bins=rms_bins,density=True)
	xlocs,binedges,nbins = binned_statistic(rms,rms,bins=rms_bins)
	return rms_dist,xlocs

def calculate_msdx_dist_scaled2(trajs,tau,corrt,tmeasure_range,data_len):
	
	###Calculate msds 

	trajs_sub = trajs[tmeasure_range,:,:]
	disps0 = trajs_sub[1:,:,:] - trajs_sub[:-1,:,:]
	disps = trajs_sub[tau:,:,:] - trajs_sub[:-1*tau,:,:]
	mean_speeds = numpy.mean(numpy.sqrt(numpy.sum(disps0**2,axis=2)),axis=0)
	rms = numpy.abs((disps[:,:,0]/(mean_speeds*numpy.sqrt(corrt))).flatten())
	rms_bins = numpy.percentile(rms,numpy.arange(0,101,int(100/data_len)))
	rms_dist,binedges = numpy.histogram(rms,bins=rms_bins,density=True)
	xlocs,binedges,nbins = binned_statistic(rms,rms,bins=rms_bins)
	return rms_dist,xlocs

def calculate_msdx_taulist(trajs,tau_list,tmeasure_range):
	trajs_sub = trajs[tmeasure_range,:,:]
	msd_list = []
	for tau in tau_list:
		disps = trajs_sub[tau:,:,:] - trajs_sub[:-1*tau,:,:]
		msd_x = numpy.mean(disps[:,:,0]**2)
		msd_list.append(msd_x)	
	return msd_list

def calculate_msdx_taulist_by_cell(trajs,tau_list,tmeasure_range):
	trajs_sub = trajs[tmeasure_range,:,:]
	msd_list = []
	for tau in tau_list:
		disps = trajs_sub[tau:,:,:] - trajs_sub[:-1*tau,:,:]
		msd_x = numpy.mean(disps[:,:,0]**2,axis=0)
		msd_list.append(msd_x)	
	return msd_list
	
def calculate_speeds_by_cell(trajs,tau,tmeasure_range):

	trajs_sub = trajs[tmeasure_range,:,:]
	disps = trajs_sub[tau:,:,:] - trajs_sub[:-1*tau,:,:]
	
	return numpy.mean( numpy.sqrt(numpy.sum( disps**2, axis=2 )), axis=0 )

def calculate_psd(vx_vals):
	powerx_list = []
	ttot = 400
	T = 400
	len_pvec = int(T/2)
	for i in range(len(vx_vals)):
					
		vx_fft = numpy.fft.fft(vx_vals[i])
			
		powerx = numpy.zeros((len_pvec,),dtype='float')
		
		powerx[0] = 1/ttot**2*2*ttot/T*numpy.absolute(vx_fft[0])**2
		powerx[len_pvec-1] = 1/ttot**2*2*ttot/T*numpy.absolute(vx_fft[len_pvec-1])**2
				
		for k in range(1,len_pvec-1):
			powerx[k] = 1/ttot**2*ttot/T*(numpy.absolute(vx_fft[k])**2 + numpy.absolute(vx_fft[ttot-k])**2)
		
		powerx_list.append(powerx)
	powerx_arr = numpy.array(powerx_list)
	
	return numpy.sum(powerx_arr, axis=0)