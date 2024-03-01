import firedrake as fd

nensemble = [5]*5

mesh = fd.UnitIntervalMesh(1,  name="meshA")

V = fd.FunctionSpace(mesh, "CG", 1)
f = fd.Function(V, name="f")

with fd.CheckpointFile("example.h5", 'w') as afile:
    afile.save_mesh(mesh) 
    for i in range(sum(nensemble)):
        q = fd.Constant(1.0)
        f.interpolate(q*i)
        print(fd.norm(f))
        afile.save_function(f, idx = i)