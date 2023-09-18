from firedrake import *
import numpy as np

n = 100
nstep = 50

mesh = PeriodicIntervalMesh(n, 40.0)
x, = SpatialCoordinate(mesh)
V = FunctionSpace(mesh, "CG", 1)
W_F = FunctionSpace(mesh, "DG", 0) 
dW = Function(W_F) 

alpha_w = CellVolume(mesh)

dphi = TestFunction(V)
du = TrialFunction(V)
        
# dU = Function(V)
kappa_isq = 0.01
a_w = (dphi*du + kappa_isq*dphi.dx(0)*du.dx(0))*dx
sp={'ksp_type': 'preonly', 'pc_type': 'lu'}

du = Function(V)

step = 0
ufile = File('dW_fig/du.pvd')
ufile.write(du, time=step)
all_us = []


for step in range(nstep):
    print("Step", step)
    dW = Function(W_F) 
    pcg = PCG64()
    rg = RandomGenerator(pcg)
    dW.assign(rg.normal(W_F, 0., 0.125))
    L_w = alpha_w*dphi*dW*dx
    solve(a_w == L_w, du, solver_parameters=sp)
    du.assign(du)
    print('dW', dW.dat.data)
    print('dU', du.dat.data)
    # store fig
    ufile.write(du, time=step)
    all_us.append(Function(du))


