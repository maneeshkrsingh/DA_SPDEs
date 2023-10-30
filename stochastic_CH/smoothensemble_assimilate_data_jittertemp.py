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

""" read obs from saved file 
    Do assimilation step for tempering and jittering steps 
"""
nsteps = 5
xpoints = 40 # no of weather station
model = Camsholm(100, nsteps, xpoints, seed = 389, lambdas=True)

MALA = False
verbose = True
nudging = True

jtfilter = jittertemp_filter(n_jitt = 0, delta = 0.025,
                              verbose=verbose, MALA=MALA,
                              nudging=nudging, visualise_tape=False)

# jtfilter = bootstrap_filter(verbose=verbose)

nensemble = [4]*25
jtfilter.setup(nensemble, model)

x, = SpatialCoordinate(model.mesh) 

# # double elliptic problem to have smoother initial conditions in space
p = TestFunction(model.V)
q = TrialFunction(model.V)
xi = Function(model.V) # To insert noise 
a = inner(grad(p), grad(q))*dx + p*q*dx
L_1 = p*abs(xi)*dx
dW_1 = Function(model.V) # For soln vector
dW_prob_1 = LinearVariationalProblem(a, L_1, dW_1)
dw_solver_1 = LinearVariationalSolver(dW_prob_1,
                                         solver_parameters={'mat_type': 'aij', 'ksp_type': 'preonly','pc_type': 'lu'})

L_2 = p*dW_1*dx
dW_2 = Function(model.V) # For soln vector
dW_prob_2 = LinearVariationalProblem(a, L_2, dW_2)
dw_solver_2 = LinearVariationalSolver(dW_prob_2,
                                         solver_parameters={'mat_type': 'aij', 'ksp_type': 'preonly','pc_type': 'lu'})

L_3 = p*dW_2*dx
dW_3 = Function(model.V) # For soln vector
dW_prob_3 = LinearVariationalProblem(a, L_3, dW_3)
dw_solver_3 = LinearVariationalSolver(dW_prob_3,
                                         solver_parameters={'mat_type': 'aij', 'ksp_type': 'preonly','pc_type': 'lu'})

# all_particle = np.zeros((xpoints, 4))

dx0 = model.rg.normal(model.R, 0.0, 1.0)
dx1 = model.rg.normal(model.R, 0.0, 0.05)
a = model.rg.normal(model.R, 0.0, 1.0)
b = model.rg.normal(model.R, 0.0, 0.05)
xi.assign(model.rg.normal(model.V, 0., 1.0))
dw_solver_1.solve()
dw_solver_2.solve()
dw_solver_3.solve()





for i in range(nensemble[jtfilter.ensemble_rank]):
    # dx1 = model.rg.normal(model.R, 0.0, 0.05)
    # b = model.rg.normal(model.R, 0.0, 0.05)
   
    # dx0 = model.rg.normal(model.R, 0.0, 1.0)

    # a = model.rg.normal(model.R, 0.0, 1.0)

    # xi.assign(model.rg.normal(model.V, 0., 1.0))
    # dw_solver_1.solve()
    # dw_solver_2.solve()
    # dw_solver_3.solve()




    # print(dW_2.dat.data[:].shape)
    _, u = jtfilter.ensemble[i][0].split()
    #print(a.dat.data , dx0.dat.data)
    u.assign(dW_3)
    # all_particle[:,i] = u.dat.data[:]




# plt.plot( all_particle[:,:])
# #plt.plot(dU.dat.data[:], 'b-', label='true soln')
# plt.legend()
# # plt.plot(dw_std, 'r-')
# plt.show()


def log_likelihood(y, Y):
    ll = (y-Y)**2/0.25**2/2*dx
    #ll = (y-Y)**2*dx
    return ll
    
#Load data
# y_exact = np.load('../../DA_Results/y_true.npy')
y = np.load('../../DA_Results/y_obs.npy') 
N_obs = y.shape[0]

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

ys = y.shape
if COMM_WORLD.rank == 0:
    y_e = np.zeros((np.sum(nensemble), ys[0], ys[1]))
    y_sim_obs_alltime_step = np.zeros((np.sum(nensemble),nsteps,  ys[1]))
    y_sim_obs_allobs_step = np.zeros((np.sum(nensemble),nsteps*N_obs,  ys[1]))

    y_sim_obs_alltime_step_new = np.zeros((np.sum(nensemble),nsteps,  ys[1]))
    y_sim_obs_allobs_step_new = np.zeros((np.sum(nensemble),nsteps*N_obs,  ys[1]))
    #print(np.shape(y_sim_obs_allobs_step))





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
        # model.randomize(jtfilter.proposal_ensemble[i])
        # z = model.forcast_data_allstep(jtfilter.proposal_ensemble[i])
        # #PETSc.Sys.Print(np.array(z).shape)
        # for step in range(nsteps):
        #     for m in range(y.shape[1]):
        #         y_sim_obs_list_new[m].dlocal[i] = z[step][m]

                
    # for step in range(nsteps):
    #     for m in range(y.shape[1]):
    #         y_sim_obs_list_new[m].synchronise()
    #         if COMM_WORLD.rank == 0:
    #             y_sim_obs_alltime_step_new[:, step, m] = y_sim_obs_list_new[m].data()
    #             y_sim_obs_allobs_step_new[:,nsteps*k+step,m] = y_sim_obs_alltime_step_new[:, step, m]


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
    #print(ESS_arr)
    print("Time shape", y_sim_obs_alltime_step.shape)
    #print("Time", y_sim_obs_alltime_step)
    print("Obs shape", y_sim_obs_allobs_step.shape)
    print("Ensemble member", y_e.shape)
    
    
    if not nudging:
        np.save("../../DA_Results/bs_ESS.npy",np.array((ESS_arr)))
        #np.save("../../DA_Results/temp.npy",np.array((temp_run_count)))
        np.save("../../DA_Results/bs_assimilated_ensemble.npy", y_e)
        np.save("../../DA_Results/bs_simualated_all_time_obs.npy", y_sim_obs_allobs_step)
        # np.save("../../DA_Results/mcmcnew_simualated_all_time_obs.npy", y_sim_obs_allobs_step_new)
    if nudging:
        np.save("../../DA_Results/withouttemp_nudge_ESS.npy",np.array((ESS_arr)))
        #np.save("../../DA_Results/Nudge_temp.npy",np.array((temp_run_count)))
        np.save("../../DA_Results/nudge_assimilated_ensemble.npy", y_e)
        np.save("../../DA_Results/nudge_simualated_all_time_obs.npy", y_sim_obs_allobs_step)
        # np.save("../../DA_Results/nudgenew_simualated_all_time_obs.npy", y_sim_obs_allobs_step_new)

# Ys_obs = np.load("simualated_all_time_obs.npy")
# Ys_obs_new = np.load("new_simualated_all_time_obs.npy")
# print(np.min(Ys_obs_new-Ys_obs))