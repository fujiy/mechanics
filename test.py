import os
os.environ['SYMPY_USE_CACHE'] = 'debug'
from sympy import *
from mechanics import System
import mechanics.integrator

system = (System()
    .add_coordinate(r'\theta', space=S)
    .add_constant('m g l')
    .define('x y', 'l * cos(theta), l * sin(theta)')
    .define('T', 'm/2 * (diff(x, t)**2 + diff(y, t)**2)')
    .define('U', 'm * g * y')
    .define('L', 'T - U')
    .euler_lagrange_equation('L')
)
# system('diff(r, t)')

system_d = (system
    .add_constant('N h')
    .add_index('i', 0, 'N')
    .discretize('i', 't', step='h')
    .apply_integrator(mechanics.integrator.Euler('h'))
)

solver = system_d.solver()