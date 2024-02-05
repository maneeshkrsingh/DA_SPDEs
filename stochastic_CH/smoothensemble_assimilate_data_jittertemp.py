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
os.makedirs('../../DA_Results/smoothDA/SALT/', exist_ok=True)

## Load data

# y_exact = np.load('../../DA_Results/y_true.npy')
y = np.load('../../DA_Results/smoothDA/SALT/y_obs.npy') 
N_obs = y.shape[0]
ys = y.shape


""" read obs from saved file 
    Do assimilation step for tempering and jittering steps 
"""
nsteps = 5
xpoints =81 # no of weather station
x_disc = 100 # no of discrete points 
model = Camsholm(100, nsteps, xpoints,noise_scale = 1.0, seed = 123456789, lambdas=True, salt=True)

MALA = False
verbose = False
nudging = False

jtfilter = jittertemp_filter(n_jitt = 5, delta = 0.15,
                              verbose=verbose, MALA=MALA,
                              nudging=nudging, visualise_tape=False)

# jtfilter = bootstrap_filter(verbose=verbose)

nensemble = [5]*30
jtfilter.setup(nensemble, model)

x, = SpatialCoordinate(model.mesh) 

# #  elliptic problem to have smoother initial conditions in space
One = Function(model.V).assign(1.0)
Area = assemble(One*dx)
cell_area = assemble(CellVolume(model.mesh)*dx)/Area
#alpha_w = 1/cell_area**0.5
#kappa_inv_sq = 2*cell_area**2
kappa_inv_sq = 1

p = TestFunction(model.V)
q = TrialFunction(model.V)
xi = Function(model.V) # To insert noise 
a = kappa_inv_sq*inner(grad(p), grad(q))*dx + p*q*dx
L_1 = (1/CellVolume(model.mesh)**0.5)*p*abs(xi)*dx
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


# #prepare to save initaililzation 
y_init_list = []
for m in range(x_disc):        # 100 total no of grids 
    y_init_shared = SharedArray(partition=nensemble, 
                                  comm=jtfilter.subcommunicators.ensemble_comm)
    y_init_list.append(y_init_shared)                            

if COMM_WORLD.rank == 0:
    y_init = np.zeros((np.sum(nensemble),  x_disc))              


for i in range(nensemble[jtfilter.ensemble_rank]):
    dx1 = model.rg.normal(model.R, 0.0, 0.05)
    b = model.rg.normal(model.R, 0.0, 0.05)
   
    dx0 = model.rg.normal(model.R, 0.0, 1.0)

    a = model.rg.normal(model.R, 0.0, 1.0)

    xi.assign(model.rg.normal(model.V, 0., 1.0))
    dw_solver_1.solve()
    dw_solver_2.solve()
    dw_solver_3.solve()


    _, u = jtfilter.ensemble[i][0].split()
    
    u.assign(abs(a)*dW_3+abs(dx0))

    obsdata = u.dat.data[:]
    for m in range(x_disc):
        y_init_list[m].dlocal[i] = obsdata[m]

for m in range(x_disc):
        y_init_list[m].synchronise()
        if COMM_WORLD.rank == 0:
            y_init[:,m] = y_init_list[m].data()


if COMM_WORLD.rank == 0:
    np.save("../../DA_Results/smoothDA/SALT/init_ensemble.npy", y_init)


def log_likelihood(y, Y):
    ll = (y-Y)**2/0.5**2/2*dx
    return ll
    


yVOM = Function(model.VVOM)

# prepare shared arrays for data
y_e_list = []
y_sim_obs_list = []
y_sim_obs_list_new = []
for m in range(y.shape[1]):        
    y_e_shared = SharedArray(partition=nensemble, 
                                  comm=jtfilter.subcommunicators.ensemble_comm)
    energy_shared = SharedArray(partition=nensemble, 
                                  comm=jtfilter.subcommunicators.ensemble_comm)
    y_sim_obs_shared = SharedArray(partition=nensemble, 
                                 comm=jtfilter.subcommunicators.ensemble_comm)
    y_e_list.append(y_e_shared)
    y_sim_obs_list.append(y_sim_obs_shared)
    y_sim_obs_list_new.append(y_sim_obs_shared)

energy_shared = SharedArray(partition=nensemble, 
                                  comm=jtfilter.subcommunicators.ensemble_comm)


if COMM_WORLD.rank == 0:
    y_e = np.zeros((np.sum(nensemble), ys[0], ys[1]))
    y_sim_obs_alltime_step = np.zeros((np.sum(nensemble),nsteps,  ys[1]))
    y_sim_obs_allobs_step = np.zeros((np.sum(nensemble),nsteps*N_obs,  ys[1]))

    y_sim_obs_alltime_step_new = np.zeros((np.sum(nensemble),nsteps,  ys[1]))
    y_sim_obs_allobs_step_new = np.zeros((np.sum(nensemble),nsteps*N_obs,  ys[1]))
    energy_array = np.zeros((np.sum(nensemble),N_obs))
    #print(np.shape(y_sim_obs_allobs_step))


# outfile = []
# for i in range(nensemble[jtfilter.ensemble_rank]):
#     idx = sum(nensemble[:jtfilter.ensemble_rank]) + i
#     #print(idx, jtfilter.ensemble_rank)
#     outfile.append(File(f"../../DA_Results/smoothDA/SALT/nudge_paraview/{idx}_output.pvd", comm=jtfilter.subcommunicators.comm))


ESS_arr = []
weights_fin = []
check_weights_fin = []
temp_run_count =[]


# to store energy
#E = Function(model.V)
E_list = []

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
        ESS_arr.append(jtfilter.ess)
        weights_fin.append(jtfilter.weights)
    # if COMM_WORLD.rank == 0:
    #     ESS_arr = []
    #     np.append(ESS_arr, jtfilter.ess)
    #     ESS_arr.append(jtfilter.ess)
    #     weights_fin.append(jtfilter.weights)
        #check_weights_fin.append(jtfilter.check_weights)
        #print('Step', k,  'Jittering accept_cout', jtfilter.accept_jitt_count)
        #temp_run_count.append(jtfilter.temp_count)
        
        
        
    for i in range(nensemble[jtfilter.ensemble_rank]):
        model.w0.assign(jtfilter.ensemble[i][0])
        #save paraview data
        _,z = model.w0.split()
        E = assemble((0.5*z*z + 0.5*z.dx(0)*z.dx(0))*dx)
        

        # outfile[i].write(z, time = k)

        obsdata = model.obs().dat.data[:]
        for m in range(y.shape[1]):
            y_e_list[m].dlocal[i] = obsdata[m]
        energy_shared.dlocal[i] = E
    E_list.append(E)
    #store_run_count = model.run_count


    energy_shared.synchronise()
    if COMM_WORLD.rank == 0:
        energy_array[:,k] = energy_shared.data()

    for m in range(y.shape[1]):
        y_e_list[m].synchronise()
        if COMM_WORLD.rank == 0:
            y_e[:, k, m] = y_e_list[m].data()




if COMM_WORLD.rank == 0:
    #print('Jittering accept_reject_cout', jtfilter.accept_reject_count)
    #print('Tempering count', temp_run_count)
    #np.save("../../DA_Results/init_ensemble.npy", y_init)
    #print(ESS_arr)
    print('energy_list', len(E_list))
    print('Average energy_array shared', np.mean(energy_array, axis=0).shape)
    print("Time shape", y_sim_obs_alltime_step.shape)

    #print("Time", y_sim_obs_alltime_step)
    print("Obs shape", y_sim_obs_allobs_step.shape)
    print("Ensemble member", y_e.shape)
    
    
    if not nudging:
        np.save("../../DA_Results/smoothDA/SALT/smooth_mcmcwt_Energy.npy",np.mean(energy_array, axis=0))
        np.save("../../DA_Results/smoothDA/SALT/smooth_mcmcwt_ESS.npy",np.array((ESS_arr)))
        np.save("../../DA_Results/smoothDA/SALT/smooth_mcmcwt_weight.npy",np.array((weights_fin)))
        #np.save("../../DA_Results/smoothDA/temp.npy",np.array((temp_run_count)))
        np.save("../../DA_Results/smoothDA/SALT/smooth_mcmcwt_assimilated_ensemble.npy", y_e)
        np.save("../../DA_Results/smoothDA/SALT/smooth_mcmcwt_simualated_all_time_obs.npy", y_sim_obs_allobs_step)
        # np.save("../../DA_Results/smoothDA/mcmcnew_simualated_all_time_obs.npy", y_sim_obs_allobs_step_new)
    if nudging:
        np.save("../../DA_Results/smoothDA/SALT/smooth_nudge_Energy.npy",np.mean(energy_array, axis=0))
        np.save("../../DA_Results/smoothDA/SALT/smooth_nudge_ESS.npy",np.array((ESS_arr)))
        np.save("../../DA_Results/smoothDA/SALT/smooth_nudge_weight.npy",np.array((weights_fin)))
        np.save("../../DA_Results/smoothDA/SALT/smooth_nudge_check_lambda_weight.npy",np.array((check_weights_fin)))
        np.save("../../DA_Results/smoothDA/nudge_temp.npy",np.array((temp_run_count)))
        np.save("../../DA_Results/smoothDA/SALT/smooth_nudge_assimilated_ensemble.npy", y_e)
        np.save("../../DA_Results/smoothDA/SALT/smooth_nudge_simualated_all_time_obs.npy", y_sim_obs_allobs_step)
        np.save("../../DA_Results/smoothDA/nudgenew_simualated_all_time_obs.npy", y_sim_obs_allobs_step_new)
