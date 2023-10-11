from firedrake import *
from nudging import *
import numpy as np
import gc
import petsc4py
import matplotlib.pyplot as plt
from firedrake.petsc import PETSc
from pyadjoint import AdjFloat

from nudging.models.stochastic_Camassa_Holm import Camsholm


""" read obs from saved file 
    Do assimilation step for tempering and jittering steps 
"""

nsteps = 5
xpoints = 40
model = Camsholm(100, nsteps, xpoints, lambdas=False)

MALA = False
verbose = False
nudging = False
# jtfilter = jittertemp_filter(n_jitt = 4, delta = 0.1,
#                               verbose=verbose, MALA=MALA,
#                               nudging=nudging, visualise_tape=False)


# jtfilter = jittertemp_filter(n_temp=4, n_jitt = 4, rho= 0.99,
#                              verbose=verbose, MALA=MALA)

jtfilter = bootstrap_filter()

# jtfilter = nudging_filter(n_temp=4, n_jitt = 4, rho= 0.999,
#                              verbose=verbose, MALA=MALA)

nensemble = [3]*32
jtfilter.setup(nensemble, model)

x, = SpatialCoordinate(model.mesh) 

# elliptic problem to have smoother initial conditions in space
p = TestFunction(model.V)
q = TrialFunction(model.V)
xi = Function(model.V)
a = inner(grad(p), grad(q))*dx + p*q*dx
L = p*xi*dx
dW = Function(model.V) # To insert noise 
dW_prob = LinearVariationalProblem(a, L, dW)
dw_solver = LinearVariationalSolver(dW_prob,
                                         solver_parameters={'mat_type': 'aij', 'ksp_type': 'preonly','pc_type': 'lu'})
for i in range(nensemble[jtfilter.ensemble_rank]):
    xi.assign(model.rg.uniform(model.V, 0., 1.0))
    dw_solver.solve()
    print(dW.dat.data[:].max())
    _, u = jtfilter.ensemble[i][0].split()
    u.assign(dW)



#prepare the initial ensemble
# for i in range(nensemble[jtfilter.ensemble_rank]):
#     dx0 = model.rg.normal(model.R, 0., 1.05)
#     dx1 = model.rg.normal(model.R, 0., 1.05)
#     a = model.rg.uniform(model.R, 0., 1.0)
#     b = model.rg.uniform(model.R, 0., 1.0)
#     u0_exp = (1+a)*0.2*2/(exp(x-403./15. - dx0) + exp(-x+403./15. + dx0)) \
#                     + (1+b)*0.5*2/(exp(x-203./15. - dx1)+exp(-x+203./15. + dx1))
    

#     _, u = jtfilter.ensemble[i][0].split()
#     u.interpolate(u0_exp)


def log_likelihood(y, Y):
    ll = (y-Y)**2/0.025**2/2*dx
    return ll
    
#Load data
y_exact = np.load('y_true.npy')
y = np.load('y_obs.npy') 
N_obs = y.shape[0]

yVOM = Function(model.VVOM)

# prepare shared arrays for data
# y_e_list = []
# y_sim_obs_list = []
# y_sim_obs_list_new = []
# for m in range(y.shape[1]):        
#     y_e_shared = SharedArray(partition=nensemble, 
#                                   comm=jtfilter.subcommunicators.ensemble_comm)
#     y_sim_obs_shared = SharedArray(partition=nensemble, 
#                                  comm=jtfilter.subcommunicators.ensemble_comm)
#     y_e_list.append(y_e_shared)
#     y_sim_obs_list.append(y_sim_obs_shared)
#     y_sim_obs_list_new.append(y_sim_obs_shared)

# ys = y.shape
# if COMM_WORLD.rank == 0:
#     y_e = np.zeros((np.sum(nensemble), ys[0], ys[1]))
#     y_sim_obs_alltime_step = np.zeros((np.sum(nensemble),nsteps,  ys[1]))
#     y_sim_obs_allobs_step = np.zeros((np.sum(nensemble),nsteps*N_obs,  ys[1]))

#     y_sim_obs_alltime_step_new = np.zeros((np.sum(nensemble),nsteps,  ys[1]))
#     y_sim_obs_allobs_step_new = np.zeros((np.sum(nensemble),nsteps*N_obs,  ys[1]))
#     #print(np.shape(y_sim_obs_allobs_step))


# mylist = []
# def mycallback(ensemble):
#    #x_obs =np.linspace(0, 40,num=self.xpoints, endpoint=False) # This is better choice
#    xpt = np.arange(0.5,40.0) # need to change the according to generate data
#    X = ensemble
#    mylist.append(X.at(xpt))



# ESS_arr = []
# temp_run_count =[]
# do assimiliation step
for k in range(N_obs):
    PETSc.Sys.Print("Step", k)
    yVOM.dat.data[:] = y[k, :]

    # #z = []
    # # make a copy so that we don't overwrite the initial condition
    # # in the next step
    # for i in  range(nensemble[jtfilter.ensemble_rank]):
    #     z = []
    #     for p in range(len(jtfilter.new_ensemble[i])):
    #         jtfilter.new_ensemble[i][p].assign(jtfilter.ensemble[i][p])
    #         jtfilter.proposal_ensemble[i][p].assign(jtfilter.ensemble[i][p])
    #     model.randomize(jtfilter.new_ensemble[i])
    #     model.randomize(jtfilter.proposal_ensemble[i])
    #     z = model.forcast_data_allstep(jtfilter.proposal_ensemble[i])
    #     #PETSc.Sys.Print(np.array(z).shape)
    #     for step in range(nsteps):
    #         for m in range(y.shape[1]):
    #             y_sim_obs_list_new[m].dlocal[i] = z[step][m]
    # for step in range(nsteps):
    #     for m in range(y.shape[1]):
    #         y_sim_obs_list_new[m].synchronise()
    #         if COMM_WORLD.rank == 0:
    #             y_sim_obs_alltime_step_new[:, step, m] = y_sim_obs_list_new[m].data()
    #             y_sim_obs_allobs_step_new[:,nsteps*k+step,m] = y_sim_obs_alltime_step_new[:, step, m]


    # Compute simulated observations using "prior" distribution
    # i.e. before we have used the observed data
    # for step in range(nsteps):
    #     for i in  range(nensemble[jtfilter.ensemble_rank]):
    #         model.run(jtfilter.new_ensemble[i], jtfilter.new_ensemble[i])
    #         # note, not safe in spatial parallel
    #         fwd_simdata = model.obs().dat.data[:]
    #         for m in range(y.shape[1]):
    #             y_sim_obs_list[m].dlocal[i] = fwd_simdata[m]


    #     for m in range(y.shape[1]):
    #         y_sim_obs_list[m].synchronise()
    #         if COMM_WORLD.rank == 0:
    #             y_sim_obs_alltime_step[:, step, m] = y_sim_obs_list[m].data()
    #             y_sim_obs_allobs_step[:,nsteps*k+step,m] = y_sim_obs_alltime_step[:, step, m]                

    jtfilter.assimilation_step(yVOM, log_likelihood)
    #PETSc.Sys.Print("forward model run count", model.run_count - store_run_count)
    # PETSc.garbage_cleanup(PETSc.COMM_SELF)
    # petsc4py.PETSc.garbage_cleanup(model.mesh._comm)
    # petsc4py.PETSc.garbage_cleanup(model.mesh.comm)

    # gc.collect()
    # if COMM_WORLD.rank == 0:
    #     # ESS_arr = []
    #     # np.append(ESS_arr, jtfilter.ess)
    #     ESS_arr.append(jtfilter.ess)
    #     #temp_run_count.append(jtfilter.temp_count)
        
        
    # for i in range(nensemble[jtfilter.ensemble_rank]):
    #     model.w0.assign(jtfilter.ensemble[i][0])
    #     obsdata = model.obs().dat.data[:]
    #     for m in range(y.shape[1]):
    #         y_e_list[m].dlocal[i] = obsdata[m]

    #store_run_count = model.run_count


    

    # for m in range(y.shape[1]):
    #     y_e_list[m].synchronise()
    #     if COMM_WORLD.rank == 0:
    #         y_e[:, k, m] = y_e_list[m].data()




# if COMM_WORLD.rank == 0:
#     #print(ESS_arr)
#     print("Time shape", y_sim_obs_alltime_step.shape)
#     #print("Time", y_sim_obs_alltime_step)
#     print("Obs shape", y_sim_obs_allobs_step.shape)
#     print("Ensemble member", y_e.shape)
#     np.save("withoutempMCMC_ESS.npy",np.array((ESS_arr)))
#     #np.save("temp.npy",np.array((temp_run_count)))
#     np.save("bs_assimilated_ensemble.npy", y_e)
#     np.save("bs_simualated_all_time_obs.npy", y_sim_obs_allobs_step)
#     np.save("bsnew_simualated_all_time_obs.npy", y_sim_obs_allobs_step_new)
#     if nudging:
#         np.save("SimplifiedNudge_ESS.npy",np.array((ESS_arr)))
#         np.save("Nudge_temp.npy",np.array((temp_run_count)))
#         np.save("Simplifiednudge_assimilated_ensemble.npy", y_e)
#         np.save("Simplifiednudge_simualated_all_time_obs.npy", y_sim_obs_allobs_step)
#         np.save("Simplifiednudge_new_simualated_all_time_obs.npy", y_sim_obs_allobs_step_new)

# Ys_obs = np.load("simualated_all_time_obs.npy")
# Ys_obs_new = np.load("new_simualated_all_time_obs.npy")
# print(np.min(Ys_obs_new-Ys_obs))