from firedrake import *
from nudging import *


from nudging import LSDEModel


T = 1.
nsteps = 10
dt = T/nsteps
A = 1.
D = 2.
# model
model = LSDEModel(A=A, D=D, nsteps=nsteps, dt=dt, lambdas=False)

nensemble = [5]*5

simfilter = sim_filter()
simfilter.setup(nensemble, model)

with CheckpointFile("example.h5",  'r') as afile:
    mesh = afile.load_mesh("meshA")
    for ilocal in range(nensemble[simfilter.ensemble_rank]):
        iglobal = simfilter.layout.transform_index(ilocal, itype='l', rtype='g')
        f = afile.load_function(mesh, "f", idx = iglobal)
        q = simfilter.ensemble[ilocal][0]
        q.interpolate(f)
        print('ilocal', ilocal, 'iglobal', iglobal, norm(q))

