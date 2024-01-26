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
xpoints =41 # no of weather station
x_disc = 100 # no of discrete points 
model = Camsholm(100, nsteps, xpoints,noise_scale = 1.0, seed = 123456789, lambdas=True, salt=True)

MALA = False
verbose = False
nudging = True

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
    a_square_val = assemble(a*a*dx)/Area

    dx0_square_val = assemble(dx0*dx0*dx)/Area
    dw_solver_1.solve()
    dw_solver_2.solve()
    dw_solver_3.solve()


    _, u = jtfilter.ensemble[i][0].split()
    
    #u.assign((abs(a)*dW_3+abs(dx0)))
    u.assign(a_square_val*dW_3+dx0_square_val) #  for the squared one 
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
    

# vertex only mesh to store true at all mesh points 
x_obs = np.linspace(0, 40, 101, endpoint=True)
x_obs_list = []
for i in x_obs:
    x_obs_list.append([i])

VOM = VertexOnlyMesh(model.mesh, x_obs_list)
VVOM = FunctionSpace(VOM, "DG", 0)

def obs_atall(X):
    _, u = X[0].split()
    Y = Function(VVOM)
    Y.interpolate(u)
    return Y   



yVOM = Function(model.VVOM)

# prepare shared arrays for data
y_e_list = []
y_sim_obs_list = []
y_sim_obs_list_new = []
y_e_allX_list = []


for m in range(y.shape[1]):        
    y_e_shared = SharedArray(partition=nensemble, 
                                  comm=jtfilter.subcommunicators.ensemble_comm)
    y_sim_obs_shared = SharedArray(partition=nensemble, 
                                 comm=jtfilter.subcommunicators.ensemble_comm)
    y_e_list.append(y_e_shared)
    y_sim_obs_list.append(y_sim_obs_shared)
    y_sim_obs_list_new.append(y_sim_obs_shared)



for m in range(101): 
    y_e_allXshared = SharedArray(partition=nensemble, 
                                  comm=jtfilter.subcommunicators.ensemble_comm)
    y_e_allX_list.append(y_e_allXshared)


if COMM_WORLD.rank == 0:
    y_e = np.zeros((np.sum(nensemble), ys[0], ys[1]))
    y_sim_obs_alltime_step = np.zeros((np.sum(nensemble),nsteps,  ys[1]))
    y_sim_obs_allobs_step = np.zeros((np.sum(nensemble),nsteps*N_obs,  ys[1]))
    y_e_allX = np.zeros((np.sum(nensemble),ys[0],  101))




# outfile = []
# for i in range(nensemble[jtfilter.ensemble_rank]):
#     idx = sum(nensemble[:jtfilter.ensemble_rank]) + i
#     #print(idx, jtfilter.ensemble_rank)
#     #outfile.append(File(f"../../DA_Results/smoothDA/SALT/bs_paraview_1000/{idx}_output.pvd", comm=jtfilter.subcommunicators.comm))
#     outfile.append(File(f"../../DA_Results/smoothDA/SALT/half_domain_mcmc_paraview/{idx}_output.pvd", comm=jtfilter.subcommunicators.comm))


ESS_arr = []
weights_fin = []
#check_weights_fin = []
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


    PETSc.garbage_cleanup(PETSc.COMM_SELF)
    petsc4py.PETSc.garbage_cleanup(model.mesh._comm)
    petsc4py.PETSc.garbage_cleanup(model.mesh.comm)

    gc.collect()
    if COMM_WORLD.rank == 0:
        ESS_arr.append(jtfilter.ess)
        weights_fin.append(jtfilter.weights)
    
        
        
        
    for i in range(nensemble[jtfilter.ensemble_rank]):
        model.w0.assign(jtfilter.ensemble[i][0])
        #save paraview data
        # _,z = model.w0.split()
        # outfile[i].write(z, time = k)
        obsdata = model.obs().dat.data[:]
        for m in range(y.shape[1]):
            y_e_list[m].dlocal[i] = obsdata[m]
        #store at all mesh points
        sim_allX = obs_atall(jtfilter.ensemble[i]).dat.data[:]
        for n in range(101):
            y_e_allX_list[n].dlocal[i] = sim_allX[n]




    

    for m in range(y.shape[1]):
        y_e_list[m].synchronise()
        if COMM_WORLD.rank == 0:
            y_e[:, k, m] = y_e_list[m].data()

    for m in range(101):
        y_e_allX_list[m].synchronise()
        if COMM_WORLD.rank == 0:
            y_e_allX[:, k, m] = y_e_allX_list[m].data()


if COMM_WORLD.rank == 0:
    print("Time shape", y_sim_obs_alltime_step.shape)
    print("Obs shape", y_sim_obs_allobs_step.shape)
    print("Ensemble member", y_e.shape)
    print("Ensemble member at all X", y_e_allX.shape)
    
    
    if not nudging:
        np.save("../../DA_Results/smoothDA/SALT/smooth_mcmcwt_ESS.npy",np.array((ESS_arr)))
        np.save("../../DA_Results/smoothDA/SALT/smooth_mcmcwt_weight.npy",np.array((weights_fin)))
        np.save("../../DA_Results/smoothDA/SALT/smooth_mcmcwt_assimilated_ensemble.npy", y_e)
        np.save("../../DA_Results/smoothDA/SALT/smooth_mcmcwt_assimilated_ensemble_allX.npy", y_e_allX)
        np.save("../../DA_Results/smoothDA/SALT/smooth_mcmcwt_simualated_all_time_obs.npy", y_sim_obs_allobs_step)

    if nudging:
        np.save("../../DA_Results/smoothDA/SALT/smooth_nudge_ESS.npy",np.array((ESS_arr)))
        np.save("../../DA_Results/smoothDA/SALT/smooth_nudge_weight.npy",np.array((weights_fin)))
        np.save("../../DA_Results/smoothDA/SALT/smooth_nudge_assimilated_ensemble.npy", y_e)
        np.save("../../DA_Results/smoothDA/SALT/smooth_nudge_assimilated_ensemble_allX.npy", y_e_allX)
        np.save("../../DA_Results/smoothDA/SALT/smooth_nudge_simualated_all_time_obs.npy", y_sim_obs_allobs_step)

