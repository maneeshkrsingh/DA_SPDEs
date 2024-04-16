import firedrake as fd
import numpy as np
from firedrake.output import VTKFile
from nudging.models.stochastic_euler import Euler_SD

import os
os.makedirs('../../DA_Results/2DEuler/', exist_ok=True)

#noise = VTKFile("../../DA_Results/2DEuler/paraview_noise/noise.pvd")

n= 64

N_time = 100

mesh = fd.UnitSquareMesh(n, n, quadrilateral = True)
model = Euler_SD(n, nsteps=5, mesh = mesh, dt = 0.0125, noise_scale=0.15)
model.setup()


x = fd.SpatialCoordinate(mesh)

dx = fd.dx

Vcg = fd.FunctionSpace(mesh, "CG", 1)  # Streamfunctions
W_F = fd.FunctionSpace(mesh, "DG", 0) # noise

sp = {"ksp_type": "cg", "pc_type": "lu",
              "pc_factor_mat_solver_type": "mumps"}

dW = fd.Function(W_F)
dW_phi = fd.TestFunction(Vcg)
dU = fd.TrialFunction(Vcg)

cell_area = fd.CellVolume(mesh)
#alpha_w = (1/cell_area**0.5)
#kappa_inv_sq = cell_area
kappa_inv_sq = fd.Constant(1/30**2)
#print(fd.assemble(kappa_inv_sq*dx(mesh)))

dU_1 = fd.Function(Vcg)
dU_2 = fd.Function(Vcg)
dU_3 = fd.Function(Vcg)

# zero boundary condition for noise
bc = fd.DirichletBC(Vcg, 0,"on_boundary")

a_dW = kappa_inv_sq*fd.inner(fd.grad(dU), fd.grad(dW_phi))*dx + dU*dW_phi*dx

L_w1 = dW*dW_phi*dx
w_prob1 = fd.LinearVariationalProblem(a_dW, L_w1, dU_1, bcs=bc, constant_jacobian=True)
wsolver1 = fd.LinearVariationalSolver(w_prob1,solver_parameters=sp)

L_w2 = dU_1*dW_phi*dx
w_prob2 = fd.LinearVariationalProblem(a_dW, L_w2, dU_2, bcs=bc, constant_jacobian=True)
wsolver2 = fd.LinearVariationalSolver(w_prob2, solver_parameters=sp)

L_w3 = dU_2*dW_phi*dx
w_prob3 = fd.LinearVariationalProblem(a_dW, L_w3, dU_3,bcs=bc, constant_jacobian=True)
wsolver3 = fd.LinearVariationalSolver(w_prob3, solver_parameters=sp)

# for t in range(N_time):
#     print('step', t)
#     dW.assign(model.rg.normal(W_F, 0., 1.0))
#     #print('source noise', fd.assemble(dW*dx))

#     wsolver1.solve()
#     wsolver2.solve() # too smooth 
#     wsolver3.solve()
#     print('max noise value', np.abs(dU_3.dat.data[:]).max())
#     #noise.write(dU_3)

Vdg = fd.FunctionSpace(mesh, "DQ", 1)  # PV space

psi = fd.TrialFunction(Vcg)
phi = fd.TestFunction(Vcg)
psi0 = fd.Function(Vcg)

q1 = fd.Function(Vdg)
q1.interpolate(fd.sin(8*fd.pi*x[0])*fd.sin(8*fd.pi*x[1])+0.4*fd.cos(6*fd.pi*x[0])*fd.cos(6*fd.pi*x[1]))

# Build the weak form for the inversion
Apsi = (fd.inner(fd.grad(psi), fd.grad(phi)))*dx
Lpsi = -q1 * phi * dx
bc1 = fd.DirichletBC(Vcg, fd.zero(), ("on_boundary"))

psi_problem = fd.LinearVariationalProblem(Apsi, Lpsi,
                                                  psi0,
                                                  bcs=bc1,
                                                  constant_jacobian=True)
sp = {"ksp_type": "cg", "pc_type": "lu",
              "pc_factor_mat_solver_type": "mumps"}
psi_solver = fd.LinearVariationalSolver(psi_problem, solver_parameters=sp)
psi_solver.solve()
dt = 0.0125
print (fd.norm(psi0))   # print the infinity norm of vector x
print(np.abs(psi0.dat.data[:]).max())
print(np.abs(dt*psi0.dat.data[:]).max())

# #to check CFL number (File :     sw_implicit/straka.py)

# DG0 = fd.FunctionSpace(mesh, "DG", 0)
# One = fd.Function(DG0).assign(1.0)
# unn = 0.5*(fd.inner(-un, n) + abs(fd.inner(-un, n))) # gives fluxes *into* cell only
# v = fd.TestFunction(DG0)
# Courant_num = fd.Function(DG0, name="Courant numerator")
# Courant_num_form = dT*(
#     both(unn*v)*(fd.dS_v + fd.dS_h)
#     + unn*v*fd.ds_tb
# )
# Courant_denom = fd.Function(DG0, name="Courant denominator")
# fd.assemble(One*v*fd.dx, tensor=Courant_denom)
# Courant = fd.Function(DG0, name="Courant")

# fd.assemble(Courant_num_form, tensor=Courant_num)
# Courant.interpolate(Courant_num/Courant_denom)