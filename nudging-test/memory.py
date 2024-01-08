from firedrake import *
from firedrake_adjoint import *
pyadjoint.tape.pause_annotation()
import petsc4py, gc

mesh = UnitSquareMesh(50, 50)
V = FunctionSpace(mesh, "CG", 1)

u = Function(V)
f = Function(V)

x, y = SpatialCoordinate(mesh)

f.interpolate(cos(2*pi*x)*sin(2*pi*y))
v = TestFunction(V)
q = Function(V)
L = inner(grad(v), grad(u - q))*dx - f*v*dx

prob = NonlinearVariationalProblem(L, u)
solver = NonlinearVariationalSolver(prob, solver_parameters=
                                    {'ksp_type':'preonly',
                                     'pc_type':'lu'})

pyadjoint.tape.continue_annotation()
solver.solve()
J = assemble(u*u*dx)
Jhat = ReducedFunctional(J, [Control(q), Control(f)], derivative_components=[0])
pyadjoint.tape.pause_annotation()

f0 = Function(V)
f0.assign(f)

for i in range(1000):
    print(i)
    f.assign(f0*(1+0.5*sin(i)))
    Jhat([q, f])
    Jhat.derivative()
    #Xopt= minimize(Jhat)
    gc.collect()
    petsc4py.PETSc.garbage_cleanup(mesh._comm)
    #print(i, Jhat(Xopt), norm(f), norm(Xopt[1]), norm(Xopt[0]))
