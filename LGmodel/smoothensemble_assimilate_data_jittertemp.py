from firedrake import *
from nudging import *
import numpy as np
import gc
import petsc4py
import matplotlib.pyplot as plt
from firedrake.petsc import PETSc
from pyadjoint import AdjFloat

from nudging.models.stochastic_Camassa_Holm import Camsholm

import os
os.makedirs('../../DA_Results/', exist_ok=True)

## Load data

# y_exact = np.load('../../DA_Results/y_true.npy')
y = np.load('../../DA_Results/w_obs.npy') 
N_obs = y.shape[0]
ys = y.shape


""" read obs from saved file 
    Do assimilation step for tempering and jittering steps 
"""
nsteps = 5
xpoints =41 # no of weather station
x_disc = 100 # no of discrete points 
model = Camsholm(100, nsteps, xpoints, seed = 123456789, lambdas=True)

MALA = False
verbose = True
nudging = False

# jtfilter = jittertemp_filter(n_jitt = 20, delta = 0.1,
#                               verbose=verbose, MALA=MALA,
#                               nudging=nudging, visualise_tape=False)

jtfilter = bootstrap_filter(verbose=verbose)

nensemble = [4]*25
jtfilter.setup(nensemble, model)

for i in range(nensemble[jtfilter.ensemble_rank]):

    u = jtfilter.ensemble[i][0]
    Q = np.random.normal(0, 0.25)
    u.assign(Q)

def log_likelihood(y, Y):
    ll = (y-Y)**2/0.25**2/2*dx
    return ll
    


yVOM = Function(model.VVOM)

# prepare shared arrays for data
y_e_list = []
y_sim_obs_list = []
y_sim_obs_list_new = []
for m in range(y.shape[1]):        
    y_e_shared = SharedArray(partition=nensemble, 
                                  comm=jtfilter.subcommunicators.ensemble_comm)
    y_sim_obs_shared = SharedArray(partition=nensemble, 
                                 comm=jtfilter.subcommunicators.ensemble_comm)
    y_e_list.append(y_e_shared)
    y_sim_obs_list.append(y_sim_obs_shared)
    y_sim_obs_list_new.append(y_sim_obs_shared)


if COMM_WORLD.rank == 0:
    y_e = np.zeros((np.sum(nensemble), ys[0], ys[1]))
    y_sim_obs_alltime_step = np.zeros((np.sum(nensemble),nsteps,  ys[1]))
    y_sim_obs_allobs_step = np.zeros((np.sum(nensemble),nsteps*N_obs,  ys[1]))


ESS_arr = []
temp_run_count =[]
# do assimiliation step
for k in range(N_obs):
    PETSc.Sys.Print("Step", k)
    yVOM.dat.data[:] = y[k, :]

    # make a copy so that we don't overwrite the initial condition
    # in the next step
    for i in  range(nensemble[jtfilter.ensemble_rank]):
        for p in range(len(jtfilter.new_ensemble[i])):
            jtfilter.new_ensemble[i][p].assign(jtfilter.ensemble[i][p])
        model.randomize(jtfilter.new_ensemble[i])
        
    # Compute simulated observations using "prior" distribution
    # i.e. before we have used the observed data
    for step in range(nsteps):
        for i in  range(nensemble[jtfilter.ensemble_rank]):
            model.run(jtfilter.new_ensemble[i], jtfilter.new_ensemble[i])
            # note, not safe in spatial parallel
            fwd_simdata = model.obs().dat.data[:]
            for m in range(y.shape[1]):
                y_sim_obs_list[m].dlocal[i] = fwd_simdata[m]


        for m in range(y.shape[1]):
            y_sim_obs_list[m].synchronise()
            if COMM_WORLD.rank == 0:
                y_sim_obs_alltime_step[:, step, m] = y_sim_obs_list[m].data()
                y_sim_obs_allobs_step[:,nsteps*k+step,m] = y_sim_obs_alltime_step[:, step, m]   


    jtfilter.assimilation_step(yVOM, log_likelihood)


    ##PETSc.Sys.Print("forward model run count", model.run_count - store_run_count)
    PETSc.garbage_cleanup(PETSc.COMM_SELF)
    petsc4py.PETSc.garbage_cleanup(model.mesh._comm)
    petsc4py.PETSc.garbage_cleanup(model.mesh.comm)

    gc.collect()
    if COMM_WORLD.rank == 0:
        ESS_arr = []
        np.append(ESS_arr, jtfilter.ess)
        ESS_arr.append(jtfilter.ess)
        #print('Step', k,  'Jittering accept_cout', jtfilter.accept_jitt_count)
        #temp_run_count.append(jtfilter.temp_count)
        
        
        
    for i in range(nensemble[jtfilter.ensemble_rank]):
        model.w0.assign(jtfilter.ensemble[i][0])
        obsdata = model.obs().dat.data[:]
        for m in range(y.shape[1]):
            y_e_list[m].dlocal[i] = obsdata[m]

    #store_run_count = model.run_count

    for m in range(y.shape[1]):
        y_e_list[m].synchronise()
        if COMM_WORLD.rank == 0:
            y_e[:, k, m] = y_e_list[m].data()

if COMM_WORLD.rank == 0:
    #print('Jittering accept_reject_cout', jtfilter.accept_reject_count)
    #print('Tempering count', temp_run_count)
    #np.save("../../DA_Results/init_ensemble.npy", y_init)
    #print(ESS_arr)
    print("Time shape", y_sim_obs_alltime_step.shape)
    #print("Time", y_sim_obs_alltime_step)
    print("Obs shape", y_sim_obs_allobs_step.shape)
    print("Ensemble member", y_e.shape)
    
    
    if not nudging:
        np.save("../../DA_Results/smooth_mcmc_ESS.npy",np.array((ESS_arr)))
        #np.save("../../DA_Results/temp.npy",np.array((temp_run_count)))
        np.save("../../DA_Results/smooth_mcmc_assimilated_ensemble.npy", y_e)
        np.save("../../DA_Results/smooth_mcmc_simualated_all_time_obs.npy", y_sim_obs_allobs_step)
        # np.save("../../DA_Results/mcmcnew_simualated_all_time_obs.npy", y_sim_obs_allobs_step_new)
    if nudging:
        np.save("../../DA_Results/smooth_nudge_ESS.npy",np.array((ESS_arr)))
        np.save("../../DA_Results/nudge_temp.npy",np.array((temp_run_count)))
        np.save("../../DA_Results/smooth_nudge_assimilated_ensemble.npy", y_e)
        np.save("../../DA_Results/smooth_nudge_simualated_all_time_obs.npy", y_sim_obs_allobs_step)
        # np.save("../../DA_Results/nudgenew_simualated_all_time_obs.npy", y_sim_obs_allobs_step_new)
