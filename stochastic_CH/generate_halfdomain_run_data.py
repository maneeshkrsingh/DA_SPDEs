from ctypes import sizeof
from fileinput import filename
from firedrake import *
from pyop2.mpi import MPI
from nudging import *
import numpy as np
import matplotlib.pyplot as plt
from nudging.models.stochastic_Camassa_Holm import Camsholm
import os

os.makedirs('../../DA_Results/smoothDA/SALT/', exist_ok=True)

"""
create some synthetic data/observation data at T_1 ---- T_Nobs
Pick initial conditon
run model, get obseravation
add observation noise N(0, sigma^2)
"""
nsteps = 5
xpoints = 41 # no of weather station

N_obs = 2000
model = Camsholm(100, nsteps, xpoints, noise_scale = 1.0, seed=1234567890, salt=True)
model.setup()
x, = SpatialCoordinate(model.mesh)

X_truth = model.allocate()
#Y_truth = model.allocate()

# # double elliptic problem to have smoother initial conditions in space
One = Function(model.V).assign(1.0)
Area = assemble(One*dx)
cell_area = assemble(CellVolume(model.mesh)*dx)/Area
#print('celllength', cell_area)
#alpha_w = 1/cell_area**0.5
#kappa_inv_sq = 2*cell_area**2
kappa_inv_sq = 1
#print(kappa_inv_sq)


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




dx0 = model.rg.normal(model.R, 0.0, 1.0)
a = model.rg.normal(model.R, 0.0, 1.)
xi.assign(model.rg.normal(model.V, 0., 1.0))
a_square_val = assemble(xi*xi*dx)/Area
dx0_square_val = assemble(dx0*dx0*dx)/Area
#print('xivalue', xi.dat.data  )
dw_solver_1.solve()
dw_solver_2.solve()
dw_solver_3.solve()


_, u = X_truth[0].split()
#u.assign((abs(a)*dW_3+abs(dx0)))
u.assign(a_square_val*dW_3+dx0_square_val)  # for the square one 


# _, u0 = Y_truth[0].split()
# u0.assign(a*dW_3)



# plt.plot(u.dat.data[:], 'b-', label='true inital')
# plt.legend()
# # plt.plot(dw_std, 'r-')
# plt.show()



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


np.save("../../DA_Results/smoothDA/SALT/y_inc_true.npy", u.dat.data[:])
# truth = File("../../DA_Results/smoothDA/SALT/half_domain_mcmc_paraview/truth.pvd")

y_true = model.obs().dat.data[:]
y_obs_full = np.zeros((N_obs, np.size(y_true)))
y_true_full = np.zeros((N_obs, np.size(y_true)))

y_obs_allmeshpoint = np.zeros((N_obs, 101))
y_true_allmeshpoint = np.zeros((N_obs, 101))


for i in range(N_obs):
    model.randomize(X_truth)
    model.run(X_truth, X_truth) # run method for every time step
    # _,z = X_truth[0].split()
    # truth.write(z)

    y_true = model.obs().dat.data[:]
    y_true_full[i,:] = y_true
    y_noise = model.rg.normal(0.0, 0.5, xpoints)  

    y_obs = y_true + y_noise   
    y_obs_full[i,:] = y_obs 

    # store at all mesh points 
    y_true_allmshpoint = obs_atall(X_truth).dat.data[:]
    y_noise_allmshpoint = model.rg.normal(0.0, 0.5, 101)
    y_true_allmeshpoint[i,:] = y_true_allmshpoint
    y_obs_allmeshpoint[i,:] = y_true_allmshpoint + y_noise_allmshpoint

print(y_true_allmshpoint.shape)
np.save("../../DA_Results/smoothDA/SALT/y_true.npy", y_true_full)
np.save("../../DA_Results/smoothDA/SALT/y_obs.npy", y_obs_full)

np.save("../../DA_Results/smoothDA/SALT/y_true_allmeshpoint.npy", y_true_allmeshpoint)
np.save("../../DA_Results/smoothDA/SALT/y_obs_allmeshpoint.npy", y_obs_allmeshpoint)


# plt.plot(y_true_allmshpoint)
# plt.show()


# y_alltime = model.obs().dat.data[:]
# y_true_all = np.zeros((nsteps, np.size(y_alltime)))
# y_obs_all = np.zeros((nsteps, np.size(y_alltime)))
# y_true_alltime = np.zeros((N_obs*nsteps, np.size(y_alltime)))
# y_obs_alltime = np.zeros((N_obs*nsteps, np.size(y_alltime)))

# print(y_obs_alltime.shape)

# for i in range(N_obs):
#     for step in range(nsteps):
#         model.randomize(X_truth)
#         model.run(X_truth, X_truth) # run method for every time step
#         y_alltrue = model.obs().dat.data[:]
    
#         y_true_all[step,:] = y_alltrue
#         y_true_alltime[nsteps*i+step,:] = y_true_all[step,:]

#         y_noise = np.random.normal(0.0, 0.25, xpoints)
#         y_obs_all[step,:] = y_alltrue + y_noise
#         y_obs_alltime[nsteps*i+step,:] = y_obs_all[step,:]


# np.save("../../DA_Results/y_true_alltime.npy", y_true_alltime)
# np.save("../../DA_Results/y_obs_alltime.npy", y_obs_alltime)

