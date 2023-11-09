from firedrake import *
import numpy as np
import matplotlib.pyplot as plt

mesh = UnitIntervalMesh(10)
V = FunctionSpace(mesh, "CG", 1)
x, = SpatialCoordinate(mesh)

One = Function(V).assign(1.0)
Area = assemble(One*dx)

print(Area)

R = FunctionSpace(mesh, "R", 0)

p = TestFunction(V)
q = TrialFunction(V)
f = Function(V)
f = Constant(1.0)


a = Constant(0.0)
b = Constant(0.0)

# bc = [DirichletBC(V, a, 1), DirichletBC(V, b, 2)]

bc1 = DirichletBC(V, a, 1)
bc2 = DirichletBC(V, b, 2)
bc = [bc1, bc2]

#kappa_in =  .5*CellVolume(mesh)
kappa_in =  10*CellVolume(mesh)

print('cell volume:', assemble(kappa_in*dx)/Area)
U = Function(V)

u = Function(V)

a = p*q*dx + kappa_in*inner(grad(p), grad(q))*dx 
L = f*p*dx

uprob = LinearVariationalProblem(a, L, U, bcs=bc)
usolve = LinearVariationalSolver(uprob,
                 solver_parameters={'mat_type': 'aij', 'ksp_type': 'preonly','pc_type': 'lu'})

usolve.solve()

u.assign(U)

plt.plot(u.dat.data[:], 'b-', label='true soln')
plt.legend()
# plt.plot(dw_std, 'r-')
plt.show()
