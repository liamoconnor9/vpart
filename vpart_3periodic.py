"""
Based off code from https://dedalus-project.readthedocs.io/en/latest/pages/examples/ivp_2d_shear_flow.html

Contributors:
    Sho Kawakami
    Liam O'Connor

Dedalus script simulating containing spherical particles rotating under given angular velocity either in a 
3D periodic domain or a 3D domain between two parallel no-slip walls. 
This script solves a 3D Navier Stokes Equation. It can be ran serially or in parallel, and uses the
built-in analysis framework to save data snapshots to HDF5 files. The
`plot_snapshots.py` script can be used to produce plots from the saved data.
The simulation should take about 10 cpu-minutes to run.

The inputs are:

    Reynolds - Reynold number
    eta - penalty parameter
    Lx, Ly, Lz - Size of domain, Lz for distance between wall(if option is on)
    Nx, Ny, Nz - Spatial discretization
    bounded - True: with wall, False: 3D periodic

The particle(s) are given fixed rotational velocity

To run and plot using e.g. 4 processes:
    $ mpiexec -n 4 python3 vpart_3periodic.py
    $ mpiexec -n 4 python3 vpart_3periodic.py snapshots/*.h5
"""

import numpy as np
import dedalus.public as d3
import logging
from datetime import datetime
logger = logging.getLogger(__name__)


# Parameters
bounded = False
Lx, Ly, Lz = 7, 7, 7
Nx, Ny, Nz = 128, 128, 128
Np=1
Reynolds = 10
Schmidt = 1
dealias = 3/2
stop_sim_time = 2
timestepper = d3.RK222
max_timestep = 1e-2
dtype = np.float64
Nframes = 100

# Bases
coords = d3.CartesianCoordinates('x', 'y', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(0, Ly), dealias=dealias)
if bounded:
    zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)
else:
    zbasis = d3.RealFourier(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)
bases = (xbasis, ybasis, zbasis)

# Fields
p = dist.Field(name='p', bases=bases)                                               #Pressure field
u = dist.VectorField(coords, name='u', bases=bases)                                 #Fluid velocity field
ud = dist.VectorField(coords, name='ud', bases=bases)
ut = dist.VectorField(coords, name='ut', bases=bases)                               #Flow time derivative field
tau_p = dist.Field(name='tau_p')                                                    #Tau field for incompressibility condition
if bounded:
    tau_u1 = dist.VectorField(coords,name='tau_u1',bases=(xbasis,ybasis))               #Tau field for BC condition
    tau_u2 = dist.VectorField(coords,name='tau_u2',bases=(xbasis,ybasis))               #Tau field for BC condition

#List of vectors, fields, vector fields indexed by particle
particlelocations = []
particlevelocities = []
particleorientations = []
forcelist = []
torquelist=[]
omegalist = []
Us = []
philist = []
for ii in range(Np):
    particlelocations.append(dist.VectorField(coords,name='pl'+str(ii)))
    particlevelocities.append(dist.VectorField(coords,name='pv'+str(ii)))
    forcelist.append(dist.VectorField(coords,name='fl'+str(ii)))
    particleorientations.append(dist.VectorField(coords,name='po'+str(ii)))
    torquelist.append(dist.VectorField(coords,name='tl'+str(ii)))
    omegalist.append(dist.Field(name='ol'+str(ii)))
    Us.append(dist.VectorField(coords,name='Us'+str(ii),bases=bases))
    philist.append(dist.Field(name='phi'+str(ii),bases=bases))

    particleorientations[ii]['g'][0]=1
    particleorientations[ii]['g'][1]=0
    particleorientations[ii]['g'][2]=0
    particlelocations[ii]['g'][0] = ii+Lx/2
    particlelocations[ii]['g'][1] = Ly/2
    particlelocations[ii]['g'][2] = Lz/2
    omegalist[ii]['g'][0] = 1


# Substitutions
nu = 1 / Reynolds
vareps = 0.1
vardel = 2.64822828*np.sqrt(vareps/Reynolds)
D = nu / Schmidt
x, y, z = dist.local_grids(xbasis, ybasis, zbasis)
ex, ey, ez = coords.unit_vector_fields(dist)

#Mask function
rvec = dist.VectorField(coords,name='rvec',bases=bases)
rvec['g'][0] = x
rvec['g'][1] = y
rvec['g'][2] = z
for ii in range(Np):
    r=np.sqrt((rvec-particlelocations[ii])@(rvec-particlelocations[ii]))
    philist[ii] = .5*(1-np.tanh(2*(r-1)/vardel))
    Us[ii] = omegalist[ii]*d3.CrossProduct(particleorientations[ii],rvec-particlelocations[ii])

#Extra definitions for no-slip boundary
if bounded:
    lift_basis = zbasis.derivative_basis(1)
    lift = lambda A: d3.Lift(A,lift_basis,-1)
    grad_u = d3.grad(u) + ez*lift(tau_u1)

# Define time derivative and 3D integration
from dedalus.core.operators import TimeDerivative
dt = lambda argy: TimeDerivative(argy)
integ = lambda argy: d3.Integrate(d3.Integrate(d3.Integrate(argy,"y") ,"z") ,"x") 


#Define Equations/Problems
if bounded:
    problem = d3.IVP([u,ut, p, tau_p,tau_u1,tau_u2]+particlelocations+particlevelocities, namespace=locals())
    #problem = d3.IVP([u,ut, p, tau_p,tau_u1,tau_u2]+particlevelocities, namespace=locals())
else:
    problem = d3.IVP([u,ut, p, tau_p]+particlelocations+particlevelocities, namespace=locals())

#Navier Stokes Equation w/out 
lhs = dt(u) + d3.Gradient(p)
rhs = - u@d3.Gradient(u)
if bounded:
    lhs+=lift(tau_u2)-nu*d3.Divergence(grad_u)
else:
    lhs-=nu*d3.Laplacian(u)


for ii in range(Np):
    rhs-= philist[ii]/vareps*(u-Us[ii]) #Sum for inhomogeneous forcing term in equation
    forcelist[ii] = integ(philist[ii]*(ut + (u - Us[ii])/vareps + u@d3.Gradient(u)))
    lhs4 = dt(particlevelocities[ii])
    rhs4 = forcelist[ii] #Mass is currently set to 1 can add ti if neccesary
    problem.add_equation((lhs4,rhs4)) #Evolution equation for velocity of each particle
    lhs3 = dt(particlelocations[ii])-particlevelocities[ii]
    rhs3 = 0
    problem.add_equation((lhs3,rhs3)) #Evolution equation for location of each particle

problem.add_equation((lhs,rhs)) #Navier-Stokes equation
problem.add_equation("dt(u) - ut = 0") #Time step
if bounded:
    problem.add_equation("trace(grad_u) + tau_p = 0") #Pressure/incompressibility condition
else:
    problem.add_equation("div(u) + tau_p = 0") #Pressure/incompressibility condition
problem.add_equation("integ(p) = 0") # Pressure gauge
if bounded:
    problem.add_equation("u(z=0) = 0")
    problem.add_equation("u(z=Lz) = 0")

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
for ii in range(Np):
    ud += (philist[ii]*Us[ii])
u = ud.evaluate().copy()

# Save files and name
name = datetime.today().strftime('%Y-%m-%d_%H-%M')
checkpoints = solver.evaluator.add_file_handler('checkpoints_IBHQ_'+name, sim_dt=stop_sim_time-max_timestep, max_writes=2, mode='overwrite')
checkpoints.add_tasks(solver.state,layout='g')
checkpoints.add_task(u, name = 'u2')
checkpoints.add_task(philist[0], name = 'phi')
checkpoints.add_task(Us[0], name = 'Usp')

snapshots = solver.evaluator.add_file_handler('snapshots_IBHQ_'+name,sim_dt =stop_sim_time/Nframes,max_writes = Nframes+1,mode='overwrite')
for force in forcelist:
    snapshots.add_task(force,name = force.name)
for particleloc in particlelocations:
    snapshots.add_task(particleloc,name = particleloc.name)
for omeg in omegalist:
    snapshots.add_task(omeg,name = omeg.name)
for orientation in particleorientations:
    snapshots.add_task(orientation,name = orientation.name)

# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.2, threshold=0.1,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property((u@ez)**2, name='w2')
flow.add_property(d3.dot(u, u), name='u2')

# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            max_w = np.sqrt(flow.max('w2'))
            max_u= np.sqrt(flow.max('u2'))
            logger.info('Iteration=%i, Time=%e, dt=%e, max(w)=%f, max(u2)=%f' %(solver.iteration, solver.sim_time, timestep, max_w, max_u))
except Exception as E:
    logger.error('Exception raised, triggering end of main loop.')
    print(E)
    raise
finally:
    solver.log_stats()

