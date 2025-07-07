from ctypes import sizeof
from fileinput import filename
from firedrake import *
from pyop2.mpi import MPI
from nudging import *
import numpy as np
import matplotlib.pyplot as plt
from firedrake.output import VTKFile
from nudging.models.stochastic_Camassa_Holm import Camsholm

import os
os.makedirs('../../DA_Results/SCH/', exist_ok=True)
particles = VTKFile("../../DA_Results/SCH/particles.pvd")

"""
create some synthetic data/observation data at T_1 ---- T_Nobs
Pick initial conditon
run model, get obseravation
add observation noise N(0, sigma^2)
"""
N_obs = 4000 # no of forward run
Nensemble = 9  # size of the ensemble
nsteps = 1
xpoints = 5000
Ld = 40.0
n = 5000  # number of points in the mesh
alpha = 1.0
alphasq = Constant(alpha**2)
resolutions = n
epsilon = Constant(0.01)  # small parameter for the peakon
peak_width=1/6
deltax = Ld / resolutions
model = Camsholm(5000, nsteps, xpoints, seed=12345,   lambdas=False, salt=False)# salt false, sflt true
model.setup()
obs_shape = model.obs().dat.data[:]
#print(obs_shape)
x, = SpatialCoordinate(model.mesh)
# ic dictionary
ic_dict = {'two_peaks': (0.2*2/(exp(x-403./15.*40./Ld) + exp(-x+403./15.*40./Ld))
                             + 0.5*2/(exp(x-203./15.*40./Ld)+exp(-x+203./15.*40./Ld))),
               'gaussian': 0.5*exp(-((x-10.)/2.)**2),
               'gaussian_narrow': 0.5*exp(-((x-10.)/1.)**2),
               'gaussian_wide': 0.5*exp(-((x-10.)/3.)**2),
               'peakon': conditional(x < Ld/2., exp((x-Ld/2)/sqrt(alphasq)), exp(-(x-Ld/2)/sqrt(alphasq))),
               'one_peak': 0.5*2/(exp(x-203./15.*40./Ld)+exp(-x+203./15.*40./Ld)),
               'proper_peak': 0.5*2/(exp(x-Ld/4)+exp(-x+Ld/4)),
               'new_peak': 0.5*2/(exp((x-Ld/4)/peak_width)+exp((-x+Ld/4)/peak_width)),
               'flat': Constant(2*pi**2/(9*40**2)),
               'fast_flat': Constant(0.1),
               'coshes': Constant(2000)*cosh((2000**0.5/2)*(x-0.75))**(-2)+Constant(1000)*cosh(1000**0.5/2*(x-0.25))**(-2),
               'd_peakon':exp(-sqrt((x-Ld/2)**2 + epsilon * deltax ** 2) / sqrt(alphasq)),
               'zero': Constant(0.0),
               'two_peakons': conditional(x < Ld/4, exp((x-Ld/4)/sqrt(alphasq)) - exp(-(x+Ld/4)/sqrt(alphasq)),
                                          conditional(x < 3*Ld/4, exp(-(x-Ld/4)/sqrt(alphasq)) - exp((x-3*Ld/4)/sqrt(alphasq)),
                                                      exp((x-5*Ld/4)/sqrt(alphasq)) - exp(-(x-3*Ld/4)/sqrt(alphasq)))),
                'twin_wide_gaussian': exp(-((x-10.)/3.)**2) + 0.5*exp(-((x-30.)/3.)**2),
               'twin_peakons': conditional(x < Ld/4, exp((x-Ld/4)/sqrt(alphasq)) + 0.5* exp((x-Ld/2)/sqrt(alphasq)),
                                           conditional(x < Ld/2, exp(-(x-Ld/4)/sqrt(alphasq)) + 0.5* exp((x-Ld/2)/sqrt(alphasq)),
                                                       conditional(x < 3*Ld/4, exp(-(x-Ld/4)/sqrt(alphasq)) + 0.5 * exp(-(x-Ld/2)/sqrt(alphasq)),
                                                                   exp((x-5*Ld/4)/sqrt(alphasq)) + 0.5 * exp(-(x-Ld/2)/sqrt(alphasq))))),
               'periodic_peakon': (conditional(x < Ld/2, 0.5 / (1 - exp(-Ld/sqrt(alphasq))) * (exp((x-Ld/2)/sqrt(alphasq))
                                                                                                + exp(-Ld/sqrt(alphasq))*exp(-(x-Ld/2)/sqrt(alphasq))),
                                                         0.5 / (1 - exp(-Ld/sqrt(alphasq))) * (exp(-(x-Ld/2)/sqrt(alphasq))
                                                                                               + exp(-Ld/sqrt(alphasq))*exp((x-Ld/2)/sqrt(alphasq))))),
               'cos_bell':conditional(x < Ld/4, (cos(pi*(x-Ld/8)/(2*Ld/8)))**2, 0.0),
               'antisymmetric': 1/(exp((x-Ld/4)/Ld)+exp((-x+Ld/4)/Ld)) - 1/(exp((Ld-x-Ld/4)/Ld)+exp((Ld+x+Ld/4)/Ld))}


###################  To generate data only for obs data points 
X_truth = model.allocate()
_, u0 = X_truth[0].split()
u0.interpolate(0.5*exp(-((x-10.)/1.)**2))






# # save data at all obs points
y_true_allpoints = np.zeros((N_obs, xpoints))

particles = np.zeros((Nensemble, N_obs, xpoints))

print(particles.shape)

for p in  ProgressBar("").iter(range(Nensemble)):
    for i in  ProgressBar("").iter(range(N_obs)):
        model.randomize(X_truth)
        model.run(X_truth, X_truth) # run method for every time step
        _,z = X_truth[0].subfunctions
        Zall= z.dat.data[:]
        #y_true_allpoints[i,:] = Zall
        particles[p, i, :] = Zall

   


# print(Zall)
# print(y)



np.save("../../DA_Results/SCH/particles_all_x.npy", particles)


