import os
os.environ['SYMPY_USE_CACHE'] = 'debug'
from sympy import *
from mechanics import System

system = (System()
    .add_index('n', 1, 2)
    .add_coordinate(r'\theta', index='n', space=S)
    .add_constant('m g')
    .add_constant('l', index='n')
    .define('x y', 'l * cos(theta), l * sin(theta)', ('n', 1))
    .define('x y', 'x[1] + l * cos(theta), y[1] + l * sin(theta)', ('n', 2))
    .define('T', 'm/2 * (diff(x, t)**2 + diff(y, t)**2)', sum_for='n')
    .define('U', 'm * g * y', sum_for='n')
    # .define('L', 'T - U')
    # .euler_lagrange_equation('L')
    # .add_constant('N')
)
# system.eval('x[2]')
# print(system.eval(system.y[2]))
# system.y.subs(system.n, 1)
x = system.x[1]
print(x.args)



# dr = system.diff('r', 't')
# print(latex(dr))