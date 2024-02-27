from ctypes import sizeof
from fileinput import filename
from firedrake import *
import firedrake as fd
import numpy as np
import matplotlib.pyplot as plt

from nudging.models.stochastic_euler import Euler_SD
import os

os.makedirs('../../DA_Results/2DEuler/', exist_ok=True)
os.makedirs('../../DA_Results/2DEuler/checkpoint_files/', exist_ok=True)

"""
create some synthetic data/observation data at T_1 ---- T_Nobs
Pick initial conditon
run model, get true value and obseravation and use paraview for viewing
add observation noise N(0, sigma^2) 
"""
#np.random.seed(138)
nensemble = [5]*20
N_obs = 2
n = 64
nsteps = 5
dt = 0.1
model = Euler_SD(n, nsteps=nsteps, dt = dt, noise_scale=0.25)
model.setup()
mesh = model.mesh
x = SpatialCoordinate(mesh)

X0_truth = model.allocate()

############################# trail ###########################################################################


q0 = X0_truth[0]

q0.interpolate(sin(8*pi*x[0])*sin(8*pi*x[1])+0.4*cos(6*pi*x[0])*cos(6*pi*x[1])) 

# q0.interpolate(sin(8*pi*x[0])*sin(8*pi*x[1])+0.4*cos(6*pi*x[0])*cos(6*pi*x[1])
#                         +0.02*sin(2*pi*x[0])+0.02*sin(2*pi*x[1])+0.3*cos(10*pi*x[0])*cos(4*pi*x[1])) 





# run model for 100 times and store inital vorticity for generating data
N_time = 100
for i in range(N_time):
    model.randomize(X0_truth)
    model.run(X0_truth, X0_truth)

X_truth = model.allocate()
X_truth[0].assign(X0_truth[0])
#print(model.mesh.hmax())
# cell_area = fd.CellVolume(model.mesh)
# h = sqrt(assemble((1/n**2)*dx(model.mesh))) 
# print('h:', h, 'cell area:', h*h)
################################### Using Matern formula ###########################################
#param
# sp = {"ksp_type": "cg", "pc_type": "lu",
#               "pc_factor_mat_solver_type": "mumps"}


# Setup noise term using Matern formula
# dW = Function(model.W_F)
# dW_phi = TestFunction(model.Vcg)
# dU = TrialFunction(model.Vcg)

# cell_area = fd.CellVolume(model.mesh)
# alpha_w = (1/cell_area**0.5)
# kappa_inv_sq = fd.Constant(1.0)


# dU_1 = fd.Function(model.Vcg)
# dU_2 = fd.Function(model.Vcg)
# dU_3 = fd.Function(model.Vcg)

# #bcs_dw = fd.DirichletBC(model.Vcg,  fd.zero(), ("on_boundary"))
# a_dW = kappa_inv_sq*fd.inner(fd.grad(dW_phi),fd.grad(dU))*dx  + dW_phi* dU*dx
# L_w1 = alpha_w*dW*dW_phi*dx
# w_prob1 = fd.LinearVariationalProblem(a_dW, L_w1, dU_1)
# wsolver1 = fd.LinearVariationalSolver(w_prob1, solver_parameters=sp)

# L_w2 = alpha_w*dU_1*dW_phi*dx
# w_prob2 = fd.LinearVariationalProblem(a_dW, L_w2, dU_2)
# wsolver2 = fd.LinearVariationalSolver(w_prob2, solver_parameters=sp)

# L_w3 = alpha_w*dU_2*dW_phi*dx
# w_prob3 = fd.LinearVariationalProblem(a_dW, L_w3, dU_3)
# wsolver3 = fd.LinearVariationalSolver(w_prob3,  solver_parameters=sp)


# dW.assign(model.rg.normal(model.W_F, 0., 1.0))

# dU= fd.Function(model.Vcg)
# wsolver1.solve()
# wsolver2.solve()
# wsolver3.solve()

# dU.assign(dU_3)

# noise = File("../../DA_Results/2DEuler/paraview/noise.pvd")
# noise.write(dU)

# # plt.plot(dU.dat.data[:], 'b-', label='true soln')
# # plt.legend()
# # # plt.plot(dw_std, 'r-')
# # plt.show()


# q0_init = X_truth[0]
# q0_init.interpolate(dU_3)

##########################################################################

# #To store velocity values 
v_true = model.obs().dat.data[:]

v1_true = v_true[:,0]
v2_true = v_true[:,1]

u1_true_all = np.zeros((N_obs, np.size(v1_true)))
u2_true_all = np.zeros((N_obs, np.size(v2_true)))
u1_obs_all = np.zeros((N_obs, np.size(v1_true)))
u2_obs_all = np.zeros((N_obs, np.size(v2_true)))

# def obs(self):
#     self.q1.assign(self.q0)  # assigned at time t+1
#     self.psi_solver.solve()  # solved at t+1 for psi
#     u = self.gradperp(self.psi0)  # evaluated velocity at time t+1
#     Y = fd.Function(self.VVOM)
#     Y.interpolate(u)
#     return Y

model.q1.rename("Potential vorticity")
model.psi0.rename("Stream function")
Vu = VectorFunctionSpace(mesh, "DQ", 0)  # DQ elements for velocity
v = Function(Vu, name="gradperp(stream function)")

#v = Function(Vu)

truth = File("../../DA_Results/2DEuler/paraview_next/truth.pvd")
truth.write(model.q1, model.psi0, v)
u_energy = []
# Exact numerical approximation 
for i in range(N_obs):
    #print('step', i)
    model.randomize(X_truth)
    model.run(X_truth, X_truth)
    model.q1.assign(X_truth[0])
    model.psi_solver.solve()  # solved at t+1 for psi
    v.project(model.gradperp(model.psi0))
    print(norm(v))
    u_energy.append(norm(v))
    truth.write(model.q1, model.psi0, v)



    u_VOM = model.obs()
   
    u = u_VOM.dat.data[:]

    u1_true_all[i,:] = u[:,0]
    u2_true_all[i,:] = u[:,1]

    u_1_noise = np.random.normal(0.0, 0.25, (n+1)**2 ) # mean = 0, sd = 0.05
    u_2_noise = np.random.normal(0.0, 0.25, (n+1)**2 ) 

    u1_obs = u[:,0] + u_1_noise
    u2_obs = u[:,1] + u_2_noise


    u1_obs_all[i,:] = u1_obs
    u2_obs_all[i,:] = u2_obs

u_true_all = np.stack((u1_true_all, u2_true_all), axis=-1)
u_obs_all = np.stack((u1_obs_all, u2_obs_all), axis=-1)


u_Energy = np.array((u_energy))
np.save("../../DA_Results/2DEuler/u_true_data_new.npy", u_true_all)
np.save("../../DA_Results/2DEuler/u_obs_data_new.npy", u_obs_all)
np.save("../../DA_Results/2DEuler/u_energy_new.npy", u_Energy)


