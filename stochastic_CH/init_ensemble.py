from firedrake import *
import numpy as np
import matplotlib.pyplot as plt

n_particle = 25
n = 500
# diffusion term 
mu = 1
mesh = PeriodicIntervalMesh(n, 40.0)
V = FunctionSpace(mesh, "CG", 1)
x, = SpatialCoordinate(mesh)
u0 = Function(V)
u0.interpolate(0.2*2/(exp(x-403./15.) + exp(-x+403./15.))
               + 0.5*2/(exp(x-203./15.)+exp(-x+203./15.)))
print(u0.dat.data[:].shape)
# PCG64 random number generator
pcg = PCG64(seed=389)
rg = RandomGenerator(pcg)
# # elliptic problem to have smoother initial conditions in space
p = TestFunction(V)
q = TrialFunction(V)
xi = Function(V) # To insert noise 
#xi.assign(rg.normal(V, 0., .5))
a = mu*inner(grad(p), grad(q))*dx + p*q*dx
L_1 = p*abs(xi)*dx
dW_1 = Function(V) # For soln vector
dW_prob_1 = LinearVariationalProblem(a, L_1, dW_1)
dw_solver_1 = LinearVariationalSolver(dW_prob_1,
                                         solver_parameters={'mat_type': 'aij', 'ksp_type': 'preonly','pc_type': 'lu'})
#dw_solver_1.solve()


# second elliptic smoother
L_2 = p*dW_1*dx
dW_2 = Function(V) # For soln vector

dW_prob_2 = LinearVariationalProblem(a, L_2, dW_2)
dw_solver_2 = LinearVariationalSolver(dW_prob_2,
                                         solver_parameters={'mat_type': 'aij', 'ksp_type': 'preonly','pc_type': 'lu'})

L_3 = p*dW_2*dx
dW_3 = Function(V) # For soln vector

dW_prob_3 = LinearVariationalProblem(a, L_3, dW_3)
dw_solver_3 = LinearVariationalSolver(dW_prob_3,
                                         solver_parameters={'mat_type': 'aij', 'ksp_type': 'preonly','pc_type': 'lu'})

#dw_solver_2.solve()

dU = Function(V)
dU_rand = Function(V)

# print(type(dW_1.dat.data[:]))
# #u.assign(dW)

# # plt.plot(dW_1.dat.data[:], 'r-')
# plt.plot(dW_2.dat.data[:], 'g-')
# # plt.plot(u0.dat.data[:], 'b-')
# plt.show()
R = FunctionSpace(mesh, "R", 0)
dw_list = []
ufile = File('Particle_fig/u.pvd')
all_particle = np.zeros((n, n_particle))
b = rg.normal(R, 0., 0.5)
dx1 = rg.normal(R, 0., 0.5)
dx0 = rg.normal(R, 0.0, 1.0)
a = rg.normal(R, 0.0, 1.0)
xi.assign(rg.normal(V, 0., 1))
dw_solver_1.solve()
dw_solver_2.solve()
dw_solver_3.solve()
for i in range(n_particle):
    # b = rg.normal(R, 0., 0.5)
    # dx1 = rg.normal(R, 0., 0.5)
    # dx0 = rg.normal(R, 0.0, 1.0)

    # a = rg.normal(R, 0.0, 1.0)

    # xi.assign(rg.normal(V, 0., 1))
    # dw_solver_1.solve()
    # dw_solver_2.solve()
    # dw_solver_3.solve()
    dU_rand.assign((a+1)*dW_3+dx0+1)
    all_particle[:,i] = dU_rand.dat.data[:]
    dw_list.append(dU_rand.dat.data[:])
    ufile.write(dU_rand, time=i)

# single realization
xi.assign(rg.normal(V, 0., 1.0))
dx0 = rg.normal(R, 0.0, 1.0)

a = rg.normal(R, 0.0, 1.0)
dw_solver_1.solve()
dw_solver_2.solve()
dw_solver_3.solve()
dU.assign((a+1)*dW_3+dx0+1)

print(dx0.dat.data)

dw_nparray = np.array(dw_list)
dw_mean = np.mean(dw_nparray)
dw_std = np.std(dw_nparray)
dw_var = np.var(dw_nparray)
dw_all = np.transpose(dw_nparray, (1,0))
print(dw_mean,dw_std, dw_var)
plt.plot(all_particle[:,:], 'y-')
plt.plot(dU.dat.data[:], 'b-', label='true soln')
plt.legend()
# plt.plot(dw_std, 'r-')
plt.show()

# print(dw_mean)