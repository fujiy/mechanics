import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from mechanics.system import System
from mechanics.solver import Result
from mechanics.util import python_name

class ResultAxes(Axes):
    def __init__(self, *args, result: Result, **kwargs):
        super().__init__(*args, **kwargs)
        self._result = result
        self._system = result.system
 
    def plot(self, *args, **kwargs):
        x_lim = None
        if len(args) == 1:
            x_name = 't'
            x_lim  = (self._result[x_name][0], self._result[x_name][-1])
            y_name = args[0]
        elif len(args) == 2:
            x_name = args[0]
            y_name = args[1]
        else:
            raise ValueError('Invalid number of arguments')

        x = self._result[x_name]
        y = self._result[y_name]

        super().set_xlabel('$' + self._system.latex(python_name(x_name)) + '$')
        super().set_ylabel('$' + self._system.latex(python_name(y_name)) + '$')
        if x_lim:
            super().set_xlim(*x_lim)

        return super().plot(x, y, **kwargs)