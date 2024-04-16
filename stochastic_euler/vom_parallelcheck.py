from ctypes import sizeof
from fileinput import filename
from firedrake import *
import firedrake as fd
from pyop2.mpi import MPI
import numpy as np
import matplotlib.pyplot as plt
from firedrake.petsc import PETSc




n = 16

comm=MPI.COMM_WORLD

mesh = fd.UnitSquareMesh(n, n, quadrilateral = True, comm=comm, name ="mesh2d_per")
x, y = fd.SpatialCoordinate(mesh)

V = fd.FunctionSpace(mesh, "CG", 2)
f = fd.Function(V).interpolate(x+y**2)

x_point = np.linspace(0.0, 1, 5)
y_point = np.linspace(0.0, 1, 5)
xv, yv = np.meshgrid(x_point, y_point)
x_obs_list = np.vstack([xv.ravel(), yv.ravel()]).T.tolist()
# points = [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]]]


x_point = np.linspace(0.0, 1, 3)
y_point = np.linspace(0.0, 1, 3)
xv, yv = np.meshgrid(x_point, y_point)
points = np.vstack([xv.ravel(), yv.ravel()]).T.tolist()

VOM = fd.VertexOnlyMesh(mesh, x_obs_list, redundant=True)
VVOM = fd.FunctionSpace(VOM, "DG", 0)

f_at_points = fd.assemble(fd.interpolate(f, VVOM))

print('VVOM ===============', f_at_points.dat.data_ro)

VVOM_out = fd.FunctionSpace(VOM.input_ordering, "DG", 0)

f_at_input_points = fd.Function(VVOM_out)
f_at_input_points.dat.data_wo[:] = np.nan
f_at_input_points.interpolate(f_at_points)

print('VOM_inputorder ==============', f_at_input_points.dat.data_ro)  # will print the values at the input points















# VOM_other = fd.VertexOnlyMesh(mesh, points, redundant=False)

# VVOM_other = fd.FunctionSpace(VOM_other, "DG", 0)

# f_at_points = fd.assemble(fd.interpolate(f, VVOM_other ))

# print('VVOM_other', f_at_points.dat.data_ro)
# VVOM_outordering = fd.FunctionSpace(VOM_other.input_ordering, "DG", 0)
# f_external = fd.Function(VVOM_outordering)
# f_external.dat.data_wo[:] = points
# f_pointdta = fd.assemble(fd.interpolate(f_external, VVOM_other))
# print(f_pointdta.shape)
