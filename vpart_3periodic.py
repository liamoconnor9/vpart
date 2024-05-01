"""
Based off code from https://dedalus-project.readthedocs.io/en/latest/pages/examples/ivp_2d_shear_flow.html

Contributors:
    Sho Kawakami
    Liam O'Connor

Dedalus script simulating a 3D periodic domain containing spherical particles 
under given velocity. This script solves a 3D Navier Stokes Equation.
It can be ran serially or in parallel, and uses the
built-in analysis framework to save data snapshots to HDF5 files. The
`plot_snapshots.py` script can be used to produce plots from the saved data.
The simulation should take about 10 cpu-minutes to run.

The inputs are:

    Reynolds - Reynold number
    eta - penalty parameter

The particle(s) are given fixed rotational velocity

To run and plot using e.g. 4 processes:
    $ mpiexec -n 4 python3 vpart_3periodic.py
    $ mpiexec -n 4 python3 vpart_3periodic.py snapshots/*.h5
"""

import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)


# Parameters
Lx, Ly, Lz = 15, 15, 15
Nx, Ny, Nz = 32, 32, 32
Np=1
Reynolds = 10
Schmidt = 1
dealias = 3/2
stop_sim_time = 1
timestepper = d3.RK222
max_timestep = 1e-2
dtype = np.float64

# Bases
coords = d3.CartesianCoordinates('x', 'y', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(0, Ly), dealias=dealias)
zbasis = d3.RealFourier(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)
bases = (xbasis, ybasis, zbasis)

# Fields
p = dist.Field(name='p', bases=bases)                                               #Pressure field
u = dist.VectorField(coords, name='u', bases=bases)                                 #Fluid velocity field
ut = dist.VectorField(coords, name='ut', bases=bases)                               #Flow time derivative field
tau_p = dist.Field(name='tau_p')                                                    #Tau field for incompressibility condition

#List of values indexed by particle
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


# Substitutions
nu = 1 / Reynolds
vareps = 0.02
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


# Define problem/equations
#problem.add_equation("ut + grad(p) - nu*lap(u) = -phi/vareps*(u - Us) - u@grad(u)")
#problem.add_equation("F  = integ(phi*(ut + (u - Us)/vareps + u@grad(u)))")

problem = d3.IVP([u, ut, p, tau_p]+particlelocations+particlevelocities, namespace=locals())
lhs = ut + d3.Gradient(p) - nu*d3.Laplacian(u)
rhs = - u@d3.Gradient(u)

from dedalus.core.operators import TimeDerivative
dt = lambda argy: TimeDerivative(argy)
integ = lambda argy: d3.Integrate(d3.Integrate(d3.Integrate(argy,"y") ,"z") ,"x") 

for ii in range(Np):
    rhs-= philist[ii]/vareps*(u-Us[ii]) #Sum for inhomogeneous forcing term in equation
    forcelist[ii] = integ(philist[ii]*(ut + (u - Us[ii])/vareps + u@d3.Gradient(u)))
    lhs3 = dt(particlelocations[ii])
    lhs4 = dt(particlevelocities[ii])
    rhs3 = particlevelocities[ii]
    rhs4 = forcelist[ii]
    problem.add_equation((lhs4,rhs4))
    problem.add_equation((lhs3,rhs3))
    
    #problem.add_equation("forcelist[ii] = integ(phi[ii]*(ut + (u - Us[ii])/vareps + u@grad(u)))")#Equation for force for each particle
    #problem.add_equation("dt(dt(particlelocations[ii])) - forcelist[ii]=0")#Equation to update position of each particle, #Mass is 1 for now, change later if needed

problem.add_equation((lhs,rhs)) #Navier-Stokes equation
problem.add_equation("dt(u) - ut = 0") #Time step
problem.add_equation("div(u) + tau_p = 0") #Pressure/incompressibility condition
problem.add_equation("integ(p) = 0") # Pressure gauge

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions - is no intiliazatin=0?
for ii in range(Np):
    u += (philist[ii]*Us[ii])
u = u.evaluate().copy()
#u['g'] = u.evaluate()['g'].copy()
#u['c'] = (phi*Us).evaluate()['c'].copy()

#Set initial velocity to particle speed in particle and zeros elsewhere
# u.change_scales(1)
# Us.change_scales(1)
# phi.change_scales(1)

# Background shear
#u['g'][0] = phi*Us
#1/2 + 1/2 * (np.tanh((z-0.5)/0.1) - np.tanh((z+0.5)/0.1))
# Match tracer to shear - do I need tracers
#s['g'] = u['g'][0]

# Add small vertical velocity perturbations localized to the shear layers
#u['g'][1] += 0.1 * np.sin(2*np.pi*x/Lx) * np.exp(-(z-0.5)**2/0.01)
#u['g'][1] += 0.1 * np.sin(2*np.pi*x/Lx) * np.exp(-(z+0.5)**2/0.01)
# Analysis
# snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.1, max_writes=10)
# snapshots.add_tasks(F, name='Force')

checkpoints = solver.evaluator.add_file_handler('checkpoints', sim_dt=5, max_writes=1, mode='overwrite')
checkpoints.add_tasks(solver.state)


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
            max_u = np.sqrt(flow.max('u2'))
            logger.info('Iteration=%i, Time=%e, dt=%e, max(w)=%f, max(u2)=%f' %(solver.iteration, solver.sim_time, timestep, max_w, max_u))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()

