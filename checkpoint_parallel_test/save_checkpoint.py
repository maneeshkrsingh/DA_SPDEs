import firedrake as fd

nensemble = [5]*20

mesh = fd.UnitIntervalMesh(1,  name="meshA")

V = fd.FunctionSpace(mesh, "CG", 1)
f = fd.Function(V, name="f")

p_dump = 0

with fd.CheckpointFile("example.h5", 'w') as afile:
    afile.save_mesh(mesh) 
    for i in range(sum(nensemble)*20):
        if  i % 20 == 0:
            q = fd.Constant(1.0)
            f.interpolate(q*i)
            print('pval', p_dump,'norm',  fd.norm(f))
            afile.save_function(f, idx=p_dump) # checkpoint only ranks 
            p_dump += 1