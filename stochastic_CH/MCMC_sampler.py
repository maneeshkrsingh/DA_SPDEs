from firedrake import *
from nudging import *
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


from nudging.models.stochastic_Camassa_Holm import Camsholm


""" read obs from saved file 
    Do assimilation step for tempering and jittering steps 
"""

nsteps = 5
xpoints = 40
model = Camsholm(100, nsteps, xpoints, lambdas=True)

MALA = False
verbose = False
nudging = False
mcmc_sampler = MCMC_sampler(delta = 0.1,  n_mcmc=5)


# jtfilter = jittertemp_filter(n_temp=4, n_jitt = 4, rho= 0.99,
#                              verbose=verbose, MALA=MALA)

# jtfilter = bootstrap_filter()

# jtfilter = nudging_filter(n_temp=4, n_jitt = 4, rho= 0.999,
#                              verbose=verbose, MALA=MALA)

nensemble = [4]*5
mcmc_sampler.setup(nensemble, model)

x, = SpatialCoordinate(model.mesh) 

#prepare the initial ensemble
for i in range(nensemble[mcmc_sampler.ensemble_rank]):
    dx0 = model.rg.normal(model.R, 0., 1.05)
    dx1 = model.rg.normal(model.R, 0., 1.05)
    a = model.rg.uniform(model.R, 0., 1.0)
    b = model.rg.uniform(model.R, 0., 1.0)
    u0_exp = (1+a)*0.2*2/(exp(x-403./15. - dx0) + exp(-x+403./15. + dx0)) \
                    + (1+b)*0.5*2/(exp(x-203./15. - dx1)+exp(-x+203./15. + dx1))
    

    _, u = mcmc_sampler.ensemble[i][0].split()
    u.interpolate(u0_exp)


def log_likelihood(y, Y):
    ll = (y-Y)**2/0.025**2/2*dx
    return ll
    

# make list of N_mcmc model.allocate() functions 
# then define a function for calculations 

def monitor_dist_int(n_mcmc, Y):
    Y_mcmc = []
    for i in range(n_mcmc):
        Y_mcmc.append(Y)
    mcmc_dist_num = ((Y_mcmc[n_mcmc-1]-Y_mcmc[0])**2)*dx
    mcmc_dist_den = ((Y_mcmc[0]-Y_mcmc[n_mcmc-1])**2)*dx
    
    return mcmc_dist_num, mcmc_dist_den

#Load data
y_exact = np.load('y_true.npy')
y = np.load('y_obs.npy') 
N_obs = y.shape[0]

yVOM = Function(model.VVOM)

# prepare shared arrays for data
y_e_list = []
y_sim_obs_list = []
y_sim_obs_list_new = []
for m in range(y.shape[1]):        
    y_e_shared = SharedArray(partition=nensemble, 
                                  comm=mcmc_sampler.subcommunicators.ensemble_comm)
    y_sim_obs_shared = SharedArray(partition=nensemble, 
                                 comm=mcmc_sampler.subcommunicators.ensemble_comm)
    y_e_list.append(y_e_shared)
    y_sim_obs_list.append(y_sim_obs_shared)
    y_sim_obs_list_new.append(y_sim_obs_shared)

ys = y.shape
if COMM_WORLD.rank == 0:
    y_e = np.zeros((np.sum(nensemble), ys[0], ys[1]))
    y_sim_obs_alltime_step = np.zeros((np.sum(nensemble),nsteps,  ys[1]))
    y_sim_obs_allobs_step = np.zeros((np.sum(nensemble),nsteps*N_obs,  ys[1]))

    y_sim_obs_alltime_step_new = np.zeros((np.sum(nensemble),nsteps,  ys[1]))
    y_sim_obs_allobs_step_new = np.zeros((np.sum(nensemble),nsteps*N_obs,  ys[1]))
    #print(np.shape(y_sim_obs_allobs_step))



# do assimiliation step
for k in range(N_obs):
    PETSc.Sys.Print("Step", k)
    yVOM.dat.data[:] = y[k, :]

    mcmc_sampler.assimilation_step(yVOM, log_likelihood, monitor_dist_int)
    

