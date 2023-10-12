from firedrake import *
import numpy as np
import matplotlib.pyplot as plt

n_particle = 10
n = 100
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
pcg = PCG64(seed=123456789)
rg = RandomGenerator(pcg)
# # elliptic problem to have smoother initial conditions in space
p = TestFunction(V)
q = TrialFunction(V)
xi = Function(V) # To insert noise 
#xi.assign(rg.normal(V, 0., .5))
a = mu*inner(grad(p), grad(q))*dx + p*q*dx
L_1 = p*xi*dx
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
#dw_solver_2.solve()

dU = Function(V)

# print(type(dW_1.dat.data[:]))
# #u.assign(dW)

# # plt.plot(dW_1.dat.data[:], 'r-')
# plt.plot(dW_2.dat.data[:], 'g-')
# # plt.plot(u0.dat.data[:], 'b-')
# plt.show()


ufile = File('Particle_fig/u.pvd')
for i in range(n_particle):
    xi.assign(rg.normal(V, 0., 0.25))
    dw_solver_1.solve()
    dw_solver_2.solve()
    dU.assign(dW_2)
    ufile.write(dU, time=i)