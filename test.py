from mechanics import *

system = (
    LagrangeSystem()
    .add_coordinate(r'\theta', space=S)
    .add_constant('m g l')
    .define('x y', 'l * sin(theta), - l * cos(theta)')
    .define('T', 'm/2 * (dot(x)**2 + dot(y)**2)')
    .define('U', 'm * g * y')
    .define('L', 'T - U')
    .define('E', 'T + U')
    .euler_lagrange_equation('L')
)
system.show_all()

from mechanics.integrator import *
system_d = (
    system.discretization()
    # .add_constant('h')
    # .add_constant('N', space=Z)
    .uniform_space('t', 'i', 0, 'N', 'h')
    .apply(Euler('h'))
    .doit()
)
system_d.show_all()

solver = system_d.solver()

import numpy as np
result = solver.run({
    'l': 1,
    'm': 1,
    'g': 1,
    'N': 10,
    'h': 0.01,
    'theta[0]': np.pi/4,
    'dottheta[0]': 0,
})