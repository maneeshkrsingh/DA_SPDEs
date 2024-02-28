from firedrake import *
import firedrake as fd
from nudging import *
import numpy as np
import matplotlib.pyplot as plt
from firedrake.petsc import PETSc
from pyadjoint import AdjFloat

from nudging.models.stochastic_euler import Euler_SD
import os

os.makedirs('../../DA_Results/2DEuler/', exist_ok=True)

""" read obs from saved file 
    Do assimilation step for tempering and jittering steps 
"""

n = 16
nsteps = 5
dt = 0.1
model = Euler_SD(n, nsteps=nsteps, dt = dt, noise_scale=0.25)

MALA = False
verbose = True
nudging = False
# jtfilter = jittertemp_filter(n_jitt = 2, delta = 0.1,
#                               verbose=verbose, MALA=MALA,
#                               nudging=nudging, visualise_tape=False)


jtfilter = bootstrap_filter(verbose=verbose)

# jtfilter = nudging_filter(n_temp=4, n_jitt = 2, rho= 0.99,
#                             verbose=verbose, MALA=MALA)
# Load data
u_exact = np.load('../../DA_Results/2DEuler/u_true_data_new.npy')
u_vel = np.load('../../DA_Results/2DEuler/u_obs_data_new.npy') 

nensemble = [5]*20


jtfilter.setup(nensemble, model)

x = SpatialCoordinate(model.mesh) 

# N_e = 10
# with fd.CheckpointFile("../../DA_Results/2DEuler/checkpoint_files/ensemble_init.h5", 
#                        'r') as afile:
#     mesh = afile.load_mesh("mesh2d_per")
#     for p in range(len(nensemble)):
#         f = afile.load_function(mesh, "f_chp", idx = p) # checkpoint at only ranks 



#prepare the initial ensemble

with fd.CheckpointFile("../../DA_Results/2DEuler/checkpoint_files/ensemble_init.h5", 
                       'r') as afile:
    mesh = afile.load_mesh("mesh2d_per")
    for i in range(nensemble[jtfilter.ensemble_rank]):
        f = afile.load_function(mesh, "f_chp", idx = jtfilter.ensemble_rank) # checkpoint at only ranks 
        # a = np.random.normal(0., 0.5) 
        # b = np.random.normal(0., 0.5) 
        # q0_in = a*f+b
        q = jtfilter.ensemble[i][0]
        q.interpolate(f)


# for i in range(nensemble[jtfilter.ensemble_rank]):
#     a = model.rg.normal(model.R, 0., 0.5) 
#     b = model.rg.normal(model.R, 0., 0.5)
#     q0_in = a*sin(8*pi*x[0])*sin(8*pi*x[1])+0.4*b*cos(6*pi*x[0])*cos(6*pi*x[1])\
#                 +0.02*a*sin(2*pi*x[0])+0.02*a*sin(2*pi*x[1])+0.3*b*cos(10*pi*x[0])*cos(4*pi*x[1]) 

#     # q0_in = 0.1 *a*b* sin(x[0]) * sin(x[1])
#     q = jtfilter.ensemble[i][0]
#     q.interpolate(q0_in)
   

def log_likelihood(y, Y):
    ll = (y-Y)**2/0.025**2/2*dx
    return ll


N_obs = u_vel.shape[0]
# VVOM Function
u_VOM = Function(model.VVOM) 

# prepare shared arrays for data
u1_e_list = []
u2_e_list = []
u1_sim_list = []
u2_sim_list = []

for m in range(u_vel.shape[1]):        
    u1_e_shared = SharedArray(partition=nensemble, 
                                 comm=jtfilter.subcommunicators.ensemble_comm)
    u2_e_shared = SharedArray(partition=nensemble, 
                                 comm=jtfilter.subcommunicators.ensemble_comm)
    u1_sim_shared = SharedArray(partition=nensemble, 
                                 comm=jtfilter.subcommunicators.ensemble_comm)
    u2_sim_shared = SharedArray(partition=nensemble, 
                                 comm=jtfilter.subcommunicators.ensemble_comm)
  
    u1_e_list.append(u1_e_shared)
    u2_e_list.append(u2_e_shared)
    u1_sim_list.append(u1_sim_shared)
    u2_sim_list.append(u2_sim_shared)
  


ushape = u_vel.shape
if COMM_WORLD.rank == 0:
    u1_e = np.zeros((np.sum(nensemble), ushape[0], ushape[1]))
    u2_e = np.zeros((np.sum(nensemble), ushape[0], ushape[1]))
    u1_sim_obs_alltime_step = np.zeros((np.sum(nensemble), nsteps, ushape[1]))
    u2_sim_obs_alltime_step = np.zeros((np.sum(nensemble), nsteps, ushape[1]))
    u1_sim_obs_allobs_step = np.zeros((np.sum(nensemble), nsteps*ushape[0], ushape[1]))
    u2_sim_obs_allobs_step = np.zeros((np.sum(nensemble), nsteps*ushape[0], ushape[1]))

# do assimiliation step
for k in range(N_obs):
    PETSc.Sys.Print("Step", k)
    u_VOM.dat.data[:,0] = u_vel[k,:,0]
    u_VOM.dat.data[:,1] = u_vel[k,:,1]


    # print simulated data for velocity
    for step in range(nsteps):
        for i in range(nensemble[jtfilter.ensemble_rank]):
            model.run(jtfilter.ensemble[i], jtfilter.ensemble[i])
            fwd_simdata1 = model.obs().dat.data[:][:,0]
            fwd_simdata2 = model.obs().dat.data[:][:,1]
            for m in range(u_vel.shape[1]):
                u1_sim_list[m].dlocal[i] = fwd_simdata1[m]
                u2_sim_list[m].dlocal[i] = fwd_simdata2[m]
        # store all timesteps values
        for m in range(u_vel.shape[1]):
            u1_sim_list[m].synchronise()
            u2_sim_list[m].synchronise()
            if COMM_WORLD.rank == 0:
                u1_sim_obs_alltime_step[:, step, m] = u1_sim_list[m].data()
                u1_sim_obs_allobs_step[:,nsteps*k+step,m] = u1_sim_obs_alltime_step[:, step, m]
                u2_sim_obs_alltime_step[:, step, m] = u2_sim_list[m].data()
                u2_sim_obs_allobs_step[:,nsteps*k+step,m] = u2_sim_obs_alltime_step[:, step, m]

    jtfilter.assimilation_step(u_VOM, log_likelihood)


    for i in range(nensemble[jtfilter.ensemble_rank]):
        model.q0.assign(jtfilter.ensemble[i][0])
        obsdata1 = model.obs().dat.data[:][:,0]
        obsdata2 = model.obs().dat.data[:][:,1]
        

        for m in range(u_vel.shape[1]):
            u1_e_list[m].dlocal[i] = obsdata1[m]
            u2_e_list[m].dlocal[i] = obsdata2[m]

    for m in range(u_vel.shape[1]):
        u1_e_list[m].synchronise()
        u2_e_list[m].synchronise()

        if COMM_WORLD.rank == 0:
            u1_e[:, k, m] = u1_e_list[m].data()
            u2_e[:, k, m] = u2_e_list[m].data()
            

#PETSc.Sys.Print("--- %s seconds ---" % (time.time() - start_time))

if COMM_WORLD.rank == 0:
    u_e = np.stack((u1_e,u2_e), axis = -1)
    u_sim_allobs_step = np.stack(( u1_sim_obs_allobs_step, u2_sim_obs_allobs_step), axis = -1)
    print(u_e.shape)
    print(u_sim_allobs_step.shape)
    #np.save("../../DA_Results/2DEuler/Velocity_ensemble.npy", u_e)
    #np.save("../../DA_Results/2DEuler/Velocity simualated_all_time.npy", u_sim_allobs_step)


    if not nudging:
        np.save("../../DA_Results/2DEuler/mcmcwt_assimilated_Velocity_ensemble.npy", u_e)
    if nudging:
        np.save("../../DA_Results/2DEuler/nudge_assimilated_Velocity_ensemble.npy", u_e)

