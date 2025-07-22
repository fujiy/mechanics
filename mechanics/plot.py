import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np

from mechanics.system import System
from mechanics.solver import Result, Results
from mechanics.util import is_tuple_ish, python_name

class ResultAxes(Axes):
    def __init__(self, *args, result: Result, **kwargs):
        super().__init__(*args, **kwargs)
        self._result = result
 
    def plot(self, *args, **kwargs):
        x_lim = None
        if len(args) == 1:
            x = 't'
            x_lim  = (self._result(x)[0], self._result(x)[-1])
            y = args[0]
        elif len(args) == 2:
            x = args[0]
            y = args[1]
        else:
            raise ValueError('Invalid number of arguments')

        x_value = self._result(x)
        y_value = self._result(y)

        if is_tuple_ish(x_value):
            x_value = np.array(x_value).transpose()
        if is_tuple_ish(y_value):
            y_value = np.array(y_value).transpose()

        super().set_xlabel('$' + self._result.latex(x) + '$')
        super().set_ylabel('$' + self._result.latex(y) + '$')
        if x_lim:
            super().set_xlim(*x_lim)

        return super().plot(x_value, y_value, **kwargs)
    
class ResultsAxes(Axes):
    def __init__(self, *args, results: Results, **kwargs):
        super().__init__(*args, **kwargs)
        self._results = results
 
    def plot(self, *args, **kwargs):
        x_lim = None
        if len(args) == 1:
            x = 't'
            x_lim  = (np.min([xs[0]  for xs in self._results(x)]), 
                      np.max([xs[-1] for xs in self._results(x)]))
            y = args[0]
        elif len(args) == 2:
            x = args[0]
            y = args[1]
        else:
            raise ValueError('Invalid number of arguments')

        x_value = self._results(x)
        y_value = self._results(y)

        if is_tuple_ish(x_value):
            x_value = np.array(x_value).transpose()
        if is_tuple_ish(y_value):
            y_value = np.array(y_value).transpose()

        super().set_xlabel('$' + self._results.latex(x) + '$')
        super().set_ylabel('$' + self._results.latex(y) + '$')
        if x_lim:
            super().set_xlim(*x_lim)

        return super().plot(x_value, y_value, **kwargs)