
from sympy import *
from mechanics import System

system = System('t')
(system
    .add_coordinate(r'r')\
    .add_coordinate(r'\theta', space=S)
    .add_constant('m g l')
    .define('x y', 'r * cos(theta), r * sin(theta)')
    .define('T', 'm/2 * (diff(x, t)**2 + diff(y, t)**2)')
    .define('U', 'm * g * y')
    .define('L', 'T - U')
    .euler_lagrange_equation('L')
)

dr = system.diff('r', 't')
print(dr)