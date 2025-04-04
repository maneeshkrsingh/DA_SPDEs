import firedrake as fd
from firedrake.output import VTKFile
from firedrake import *
from nudging.models.stochastic_euler import Euler_SD
import os
os.makedirs('../../DA_Results/2DmixEuler/', exist_ok=True)

truth_init = VTKFile("../../DA_Results/2DmixEuler/truth_init.pvd")
# stream_init = VTKFile("../../DA_Results/2DmixEuler/stream_init.pvd")

n = 128
Dt = 0.25
nsteps = 50
noise_scale = 10.95
r = 1.01

mesh = fd.UnitSquareMesh(n,n,quadrilateral=True)

x = fd.SpatialCoordinate(mesh)
dx = fd.dx
dS = fd.dS

def gradperp(u):
    return fd.as_vector((-u.dx(1), u.dx(0)))

# solver_parameters
sp = {"ksp_type": "cg", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}

# FE spaces
Vcg = fd.FunctionSpace(mesh, "CG", 1)  # Streamfunctions
Vdg = fd.FunctionSpace(mesh, "DQ", 1)  # PV space
V_mix = fd.MixedFunctionSpace((Vdg, Vcg))
##################################   Matern field    ###########################
W_F = fd.FunctionSpace(mesh, "DG", 0) # noise
dW = fd.Function(W_F)
dW_phi = fd.TestFunction(Vcg)
dU = fd.TrialFunction(Vcg)

cell_area = fd.CellVolume(mesh)
kappa_inv_sq = fd.Constant(1/30**2)

dU_1 = fd.Function(Vcg)
dU_2 = fd.Function(Vcg)
dU_3 = fd.Function(Vcg)
# zero boundary condition for noise
bc_w = fd.DirichletBC(Vcg, 0,"on_boundary")

a_dW = kappa_inv_sq*fd.inner(fd.grad(dU), fd.grad(dW_phi))*dx + dU*dW_phi*dx

L_w1 = dW*dW_phi*dx
w_prob1 = fd.LinearVariationalProblem(a_dW, L_w1, dU_1, bcs=bc_w, constant_jacobian=True)
wsolver1 = fd.LinearVariationalSolver(w_prob1,solver_parameters=sp)

L_w2 = dU_1*dW_phi*dx
w_prob2 = fd.LinearVariationalProblem(a_dW, L_w2, dU_2, bcs=bc_w, constant_jacobian=True)
wsolver2 = fd.LinearVariationalSolver(w_prob2, solver_parameters=sp)

L_w3 = dU_2*dW_phi*dx
w_prob3 = fd.LinearVariationalProblem(a_dW, L_w3, dU_3,bcs=bc_w, constant_jacobian=True)
wsolver3 = fd.LinearVariationalSolver(w_prob3, solver_parameters=sp)
#################################################################################
bc = fd.DirichletBC(V_mix.sub(1), fd.zero(), ("on_boundary"))
# mix function
qpsi0 = fd.Function(V_mix) # n time 
q0,psi0  = qpsi0.subfunctions

q0.interpolate(sin(8*pi*x[0])*sin(8*pi*x[1])+0.4*cos(6*pi*x[0])*cos(6*pi*x[1])
                    +0.3*cos(10*pi*x[0])*cos(4*pi*x[1]) +0.02*sin(2*pi*x[0])+ 0.02*sin(2*pi*x[1]))


qpsi1 = fd.Function(V_mix) # n+1 time 
qpsi1.assign(qpsi0)
q1, psi1 = fd.split(qpsi1)

# mid point formulation
qh = 0.5*(q1+q0)
psih = 0.5*(psi1+psi0)

# test functions
p, phi = fd.TestFunctions(V_mix)
# upwinding terms
n_F = fd.FacetNormal(mesh)
un = 0.5 * (fd.dot(gradperp(psih), n_F) + abs(fd.dot(gradperp(psih), n_F))) # calculated at at mid pont

# bilinear form 
F = (q1-q0)*p*dx + Dt*(fd.dot(fd.grad(p), -qh*gradperp(psih)) + p*r*q1+noise_scale*p*dU_3*Dt**2)*dx\
        +Dt*(fd.dot(fd.jump(p), un("+")*qh("+") - un("-")*qh("-")))*dS\
        +(fd.inner(fd.grad(psi1), fd.grad(phi)))*dx + psi1*phi*dx +  q0*phi*dx 


# timestepping solver
qphi_prob = fd.NonlinearVariationalProblem(F, qpsi1, bcs=bc)

qphi_solver = fd.NonlinearVariationalSolver(qphi_prob,solver_parameters=sp)

# To access its output
q1, psi1 = qpsi1.subfunctions

q1.rename("Vorticity")
psi1.rename("stream function")

# random number generator
pcg = fd.PCG64(seed = 1283456)
rg = fd.RandomGenerator(pcg)

for step in range(nsteps):
    print('Step', step)
    # setup for noise
    dW.assign(rg.normal(W_F, 0., 1.0))
    wsolver1.solve()
    wsolver2.solve()
    wsolver3.solve()
    # solve 
    qphi_solver.solve()
    # update
    qpsi0.assign(qpsi1)
    print('vorticity norm', fd.norm(q1), 'psi norm', fd.norm(psi1))
    truth_init.write(q1, psi1)