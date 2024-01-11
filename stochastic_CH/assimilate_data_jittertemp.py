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

os.makedirs('../../DA_Results/non-smoothDA/', exist_ok=True)

""" read obs from saved file 
    Do assimilation step for tempering and jittering steps 
"""
nsteps = 5
xpoints = 25
model = Camsholm(100, nsteps, xpoints, seed = 12345, noise_scale = 0.5, lambdas=True)
model.setup()

x_obs = np.linspace(0, 40, num=100, endpoint=False)
#print(x_obs)
x_obs_list = []
for i in x_obs:
    x_obs_list.append([i])
VOM = VertexOnlyMesh(model.mesh, x_obs_list)
VVOM = FunctionSpace(VOM, "DG", 0)
Z = Function(VVOM)


MALA = False
verbose = True
nudging = False
# jtfilter = jittertemp_filter(n_jitt = 5, delta = 0.15,
#                               verbose=verbose, MALA=MALA,
#                               nudging=nudging, visualise_tape=False)

jtfilter = bootstrap_filter(verbose=verbose)

nensemble = [5]*6
jtfilter.setup(nensemble, model)

x, = SpatialCoordinate(model.mesh) 

#prepare the initial ensemble
for i in range(nensemble[jtfilter.ensemble_rank]):
    dx0 = model.rg.normal(model.R, 0., 1.0)
    dx1 = model.rg.normal(model.R, 0., 1.0)
    a = model.rg.normal(model.R, 0., 1.0)
    b = model.rg.normal(model.R, 0., 1.0)
    u0_exp = (a)*0.2*2/(exp(x-403./15. - dx0) + exp(-x+403./15. + dx0)) \
                    + (b)*0.5*2/(exp(x-203./15. - dx1)+exp(-x+203./15. + dx1))
    

    _, u = jtfilter.ensemble[i][0].split()
    u.interpolate(u0_exp)


def log_likelihood(y, Y):
    ll = (y-Y)**2/0.05**2/2*dx
    return ll
    
#Load data
y_exact = np.load('../../DA_Results/non-smoothDA/y_true.npy')
y = np.load('../../DA_Results/non-smoothDA/y_obs.npy') 

N_obs = y.shape[0]

yVOM = Function(model.VVOM)

# prepare shared arrays for data
y_e_list = []
y_e_list_allx = []
y_sim_obs_list = []
y_sim_obs_list_new = []

#print(y.shape[1])

for m in range(100):
    y_e_shared_allx = SharedArray(partition=nensemble, 
                                  comm=jtfilter.subcommunicators.ensemble_comm)
    y_e_list_allx.append(y_e_shared_allx)


for m in range(y.shape[1]):        
    y_e_shared = SharedArray(partition=nensemble, 
                                  comm=jtfilter.subcommunicators.ensemble_comm)
    y_sim_obs_shared = SharedArray(partition=nensemble, 
                                 comm=jtfilter.subcommunicators.ensemble_comm)
    y_e_list.append(y_e_shared)
    y_sim_obs_list.append(y_sim_obs_shared)
    y_sim_obs_list_new.append(y_sim_obs_shared)

ys = y.shape
if COMM_WORLD.rank == 0:
    y_e = np.zeros((np.sum(nensemble), ys[0], ys[1]))
    y_sim_obs_alltime_step = np.zeros((np.sum(nensemble),nsteps,  ys[1]))
    y_sim_obs_allobs_step = np.zeros((np.sum(nensemble),nsteps*N_obs,  ys[1]))
    y_e_allx = np.zeros((np.sum(nensemble), ys[0], 100))

ESS_arr = []
# temp_run_count =[]
# do assimiliation step

outfile = []
for i in range(nensemble[jtfilter.ensemble_rank]):
    idx = sum(nensemble[:jtfilter.ensemble_rank]) + i
    #print(idx, jtfilter.ensemble_rank)
    outfile.append(File(f"../../DA_Results/non-smoothDA/paraview/{idx}_output.pvd", comm=jtfilter.subcommunicators.comm))



for k in range(N_obs):
    PETSc.Sys.Print("Step", k)
    yVOM.dat.data[:] = y[k, :]
   
    # make a copy so that we don't overwrite the initial condition in the next step
    for i in  range(nensemble[jtfilter.ensemble_rank]):
        for p in range(len(jtfilter.new_ensemble[i])):
            jtfilter.new_ensemble[i][p].assign(jtfilter.ensemble[i][p])
        model.randomize(jtfilter.new_ensemble[i])
       
    # Compute simulated observations using "prior" distribution i.e. before we have used the observed data
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

    # assimilation step
    jtfilter.assimilation_step(yVOM, log_likelihood)
  
    PETSc.garbage_cleanup(PETSc.COMM_SELF)
    petsc4py.PETSc.garbage_cleanup(model.mesh._comm)
    petsc4py.PETSc.garbage_cleanup(model.mesh.comm)

    gc.collect()
    if COMM_WORLD.rank == 0:
        ESS_arr.append(jtfilter.ess)
        
    for i in range(nensemble[jtfilter.ensemble_rank]):
        
        # idx = sum(nensemble[:jtfilter.ensemble_rank]) + i
        # #print(idx, jtfilter.ensemble_rank)
        # outfile = File(f"../../DA_Results/non-smoothDA/paraview/{idx}_output.pvd", comm=jtfilter.subcommunicators.comm)
        model.w0.assign(jtfilter.ensemble[i][0])

        
        _,z = model.w0.split()
        #print(k)
        outfile[i].write(z, time = k)
        obsdata = model.obs().dat.data[:]
        for m in range(y.shape[1]):
            y_e_list[m].dlocal[i] = obsdata[m]

    for m in range(y.shape[1]):
        y_e_list[m].synchronise()
        if COMM_WORLD.rank == 0:
            y_e[:, k, m] = y_e_list[m].data()

    # for i in range(nensemble[jtfilter.ensemble_rank]):
    #     model.w0.assign(jtfilter.ensemble[i][0])
    #     _,z = model.w0.split()
    #     Zall= Z.interpolate(z).dat.data[:]
    #     for m in range(100):
    #         y_e_list_allx[m].dlocal[i] = Zall[m]

    # for m in range(100):
    #     y_e_list_allx[m].synchronise()
    #     if COMM_WORLD.rank == 0:
    #         y_e_allx[:, k, m] = y_e_list_allx[m].data()

if COMM_WORLD.rank == 0:
    #print(ESS_arr)
    print("Time shape", y_sim_obs_alltime_step.shape)
    #print("Time", y_sim_obs_alltime_step)
    print("Obs shape", y_sim_obs_allobs_step.shape)
    print("Ensemble member", y_e.shape)
    
    if not nudging:
        np.save("../../DA_Results/non-smoothDA/bs_ESS.npy",np.array((ESS_arr)))
        #np.save("../../DA_Results/temp.npy",np.array((temp_run_count)))
        np.save("../../DA_Results/non-smoothDA/bs_assimilated_ensemble.npy", y_e)
        np.save("../../DA_Results/non-smoothDA/bs_simualated_all_time_obs.npy", y_sim_obs_allobs_step)
    if nudging:
        np.save("../../DA_Results/non-smoothDA/an_nudge_ESS.npy",np.array((ESS_arr)))
        #np.save("../../DA_Results/Nudge_temp.npy",np.array((temp_run_count)))
        np.save("../../DA_Results/non-smoothDA/an_nudge_assimilated_ensemble.npy", y_e)
        np.save("../../DA_Results/non-smoothDA/an_nudge_simualated_all_time_obs.npy", y_sim_obs_allobs_step)