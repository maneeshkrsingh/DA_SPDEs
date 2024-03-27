import firedrake as fd
from firedrake.output import VTKFile
from nudging.models.stochastic_euler import Euler_SD

import os
os.makedirs('../../DA_Results/2DEuler/', exist_ok=True)

noise = VTKFile("../../DA_Results/2DEuler/paraview_noise/noise.pvd")

n= 64

N_time = 100

mesh = fd.UnitSquareMesh(n, n, quadrilateral = True)
model = Euler_SD(n, nsteps=5, mesh = mesh, dt = 0.1, noise_scale=0.15)
model.setup()

dx = fd.dx

Vcg = fd.FunctionSpace(mesh, "CG", 1)  # Streamfunctions
W_F = fd.FunctionSpace(mesh, "DG", 0) # noise

sp = {"ksp_type": "cg", "pc_type": "lu",
              "pc_factor_mat_solver_type": "mumps"}

dW = fd.Function(W_F)
dW_phi = fd.TestFunction(Vcg)
dU = fd.TrialFunction(Vcg)

cell_area = fd.CellVolume(mesh)
alpha_w = (1/cell_area**0.5)
kappa_inv_sq = cell_area
#kappa_inv_sq = fd.Constant(0.01)
print(fd.assemble(kappa_inv_sq*dx(mesh)))

dU_1 = fd.Function(Vcg)
dU_2 = fd.Function(Vcg)
dU_3 = fd.Function(Vcg)

# zero boundary condition for noise
bc = fd.DirichletBC(Vcg, 0,"on_boundary")

a_dW = kappa_inv_sq*fd.inner(fd.grad(dU), fd.grad(dW_phi))*dx + dU*dW_phi*dx

L_w1 = alpha_w*dW*dW_phi*dx
w_prob1 = fd.LinearVariationalProblem(a_dW, L_w1, dU_1, bcs=bc, constant_jacobian=True)
wsolver1 = fd.LinearVariationalSolver(w_prob1,solver_parameters=sp)

L_w2 = dU_1*dW_phi*dx
w_prob2 = fd.LinearVariationalProblem(a_dW, L_w2, dU_2, bcs=bc, constant_jacobian=True)
wsolver2 = fd.LinearVariationalSolver(w_prob2, solver_parameters=sp)

L_w3 = dU_2*dW_phi*dx
w_prob3 = fd.LinearVariationalProblem(a_dW, L_w3, dU_3,bcs=bc, constant_jacobian=True)
wsolver3 = fd.LinearVariationalSolver(w_prob3, solver_parameters=sp)

for t in range(N_time):
    print('step', t)
    dW.assign(model.rg.normal(W_F, 0., 1.0))
    print(fd.assemble(dW*dx))

    wsolver1.solve()
    wsolver2.solve() # too smooth 
    wsolver3.solve()
    noise.write(dU_3)
