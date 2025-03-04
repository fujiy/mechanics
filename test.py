
from sympy import *
from mechanics import System

system = (System()
    .add_index('n', 1, 2)
    .add_coordinate(r'\theta', index='n', space=S)
    .add_constant('m g')
    .add_constant('l', index='n')
    .define('x y', 'l * cos(theta), l * sin(theta)', ('n', 1))
    .define('x y', 'x[1] + l * cos(theta), y[1] + l * sin(theta)', ('n', 2))
    # .define('T', 'm/2 * (diff(x, t)**2 + diff(y, t)**2)')
    # .define('U', 'm * g * y')
    # .define('L', 'T - U')
    # .euler_lagrange_equation('L')
    # .add_constant('N')
)
system.eval('x[2]')

# dr = system.diff('r', 't')
# print(latex(dr))