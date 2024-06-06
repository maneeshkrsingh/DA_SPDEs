from firedrake import *
from pyop2.mpi import MPI
from nudging import *
import numpy as np

# model
# multiply by A and add D
T = 1.
nsteps = 10
dt = T/nsteps
A = 1.
D = 2.
model = LSDEModel(A=A, D=D, nsteps=nsteps, dt=dt)

# solving
# dx = A*x*dt + D*dW
# integrating factor e^{-At}
# e^{-At}dx = e^{-At}*x*dt + De^{-t}dW
# we have d(e^{-At}x) = e^{-At}dx - e^{-At}x*dt
# so
# d(e^{-At}x) = De^{-At}dW
# i.e.
# x(t) = e^{At}x(0) + int_0^t D e^{A(t-s)}dW(s)

# E(int_0^t D e^{A(t-s)}dW(s)) = 0
# E[(int_0^t D e^{A(t-s)}dW(s))^2] = int_0^t D^2 e^{2A(t-s)} ds (Ito isometry)
# = (-D^2/(2A))*[e^{2A(t-s)}]_{s=0}^{s=t}
# = (D^2/(2A))*(e^{2At} - 1)
# so pi(x(t)|x(0)) ~ N(e^{at}x(0), sig^2)
# sig^2 = (D^2/2A)*(e^{2At} - 1)

# take x(0) ~ N(c, d^2)

# write y(t) = x(t) - e^{At}x(0),
# then
# z = (y(t), x(0))^T
# and z ~ N( (0, c)^T, Sigma )
# with
# Sigma = (sig^2   0)
#         (0     d^2)

# (x(t), x(0))^T = Bz with
# B = (1  exp(At))
#     (0        1)

# (x(t), x(0))^T ~ N(B(0, c)^T, B Sigma B^T)

# Sigma B^T = (sig^2   0)(1       0) = (sig^2         0)
#             (0     d^2)(exp(At) 1)   (d^2*exp(At) d^2)

# B Sigma B^T = (1 exp(At))(sig^2         0)
#               (0       1)(d^2*exp(At) d^2)

#             = (sig^2 + d^2*exp(2At) d^2*exp(At))
#               (d^2*exp(At)                  d^2)

# B(0, c)^T = (1 exp(At))(0) = c(exp(At))
#             (0       1)(c)    (      1)

# The marginal distribution for x(t) is then
# x(t) ~ N(c*exp(At), sig^2 + d^2*exp(2At))

# now we have an observation y = x(1) + e, e ~ N(0, S^2)
# Reverend Bayes
# pi(x(1)|y) propto pi(y|x(1))*pi(x(1))
# = exp( -(1/2)[(y-x(1))^2/S^2 + (x(1)-c*exp(A))^2/(sig^2+d^2*exp(2A))])
# = exp( -(1/2)[(y-x(1))^2/S^2 + (x(1)-a)^2/b^2])
# where a = c*exp(A), b = (sig^2+d^2*exp(2A))
# completing the square
# (y-x(1))^2/S^2 + (x(1)-a)^2/b^2
# = (x(1)^2-2x(1)y + y^2)/S^2 + (x(1)^2 - 2ax(1) + a^2)/b^2
# = x(1)**2(b^2 + S^2)/(b^2S^2) - 2x(1)(b^2y + S^2a)/(b^2S^2) + stuff
# = (b^2 + S^2)/(b^2S^2)[ x(1)^2 - 2x(1)*(b^2y + S^2a)/(b^2+S^2)] + stuff
# = (b^2 + S^2)/(b^2S^2)[ x(1)^2 - 2x(1)*(b^2y + S^2a)/(b^2+S^2)] + stuff
# = (b^2 + S^2)/(b^2S^2)[ x(1)^2 - (b^2y + S^2a)/(b^2+S^2)]^2 + stuff
# where stuff is anything independent of x(1)

# then
# x(1)|y ~ N((b^2y + S^2a)/(b^2+S^2), (b^2S^2)/(b^2 + S^2))

# bootstrap filter
verbose = True
bsfilter = bootstrap_filter(verbose=verbose)

nensemble = [50]*20
bsfilter.setup(nensemble, model, residual=False)

# data
y = model.obs()
y0 = 1.2
y.dat.data[:] = y0

# prepare the initial ensemble
c = 1.
d = 1.
for i in range(nensemble[bsfilter.ensemble_rank]):
    dx0 = model.rg.normal(model.R, c, d**2)
    u = bsfilter.ensemble[i][0]
    u.assign(dx0)

# initial values in a shared array
prior = SharedArray(partition=nensemble,
                        comm=bsfilter.subcommunicators.ensemble_comm)
for i in range(nensemble[bsfilter.ensemble_rank]):
    model.u.assign(bsfilter.ensemble[i][0])
    obsdata = model.obs().dat.data[:]
    prior.dlocal[i] = obsdata

prior.synchronise()
    
# observation noise standard deviation
S = 0.3
    
def log_likelihood(y, Y):
    ll = (y-Y)**2/S**2/2*dx
    return ll

bsfilter.assimilation_step(y, log_likelihood)

# results in a shared array
posterior = SharedArray(partition=nensemble,
                        comm=bsfilter.subcommunicators.ensemble_comm)
for i in range(nensemble[bsfilter.ensemble_rank]):
    model.u.assign(bsfilter.ensemble[i][0])
    obsdata = model.obs().dat.data[:]
    posterior.dlocal[i] = obsdata
posterior.synchronise()

if COMM_WORLD.rank == 0:
    prvals = prior.data()
    #import matplotlib.pyplot as pp
    #pp.subplot(1,2,1)
    #pp.hist(prvals, bins=20)
    print("prior mean", np.mean(prvals), "variance", np.var(prvals))

    pvals = posterior.data()
    #pp.subplot(1,2,2)
    #pp.hist(pvals, bins=20)
    #pp.show()
    print("posterior mean", np.mean(pvals), "variance", np.var(pvals))

    # analytical formula
    # x(1)|y ~ N((b^2y + S^2a)/(b^2+S^2), (b^2S^2)/(b^2 + S^2))
    # where a = c*exp(A), b = (sig^2+d^2*exp(2A))
    # sig^2 = (D^2/2A)*(e^{2A} - 1)
    a = c*exp(A)
    sigsq = D**2/2/A*(exp(2*A) - 1)  # t = 1
    b = sigsq + d**2*exp(2*A)
    mean = (b**2*y0 + S**2*a)/(b**2 + S**2)
    variance = b**2*S**2/(b**2 + S**2)
    print("true mean", mean, "relative error", (mean-np.mean(pvals))/mean)
    print("true variance", variance, "relative error", (variance-np.var(pvals))/variance)
