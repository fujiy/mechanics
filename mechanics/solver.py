from typing import cast, Any
import warnings
import numpy as np
import numpy
from scipy import linalg
from scipy import sparse
from tqdm import tqdm
import sympy as sp
import matplotlib.pyplot as plt
from sympy.printing.numpy import SciPyPrinter

from mechanics.system import System
from mechanics.symbol import Function, Expr, Index
from mechanics.util import python_name, name_type

class PythonPrinter(SciPyPrinter):
    # def _print_Symbol(self, expr):
    #         print(expr)
    #         name = super()._print_Symbol(expr)
    #         return name
    
    def _print_AppliedUndef(self, expr):
        if not isinstance(expr, Function):
            raise TypeError(f'Expected Symbol, got {type(expr)}, {expr}')
        if expr.args:
            return python_name(expr.name) + f'[{", ".join(map(str, expr.args))}]'
        else:
            return python_name(expr.name)
        
class Result:
    def __init__(self, system: System):
        self.system = system

        self.newton_converged_iters = []
        self._dict = {}

    def set_data(self, keys, values, N=None):
        if N:
            self._dict.update({python_name(key.name): value[:N] for key, value in zip(keys, values)})
        else:
            self._dict.update({python_name(key.name): value for key, value in zip(keys, values)})

    # Access
    
    def __getattr__(self, name: str) -> np.ndarray:
        if name in self._dict: return self._dict[name]
        else:                  raise AttributeError(f'\'{name}\' is not exists')
        
    def __getitem__(self, name: str) -> np.ndarray:
        name = python_name(name)
        if name in self._dict: return self._dict[name]
        else:                  raise KeyError(f'\'{name}\' is not exists')

    def __contains__(self, name: str) -> bool:
        name = python_name(name)
        return name in self._dict
    
    # For plotting
    def _as_mpl_axes(self) -> Any:
        from mechanics.plot import ResultAxes
        return ResultAxes, { 'result': self } 
        

class Solver:

    variables: tuple[Function, ...]

    def __init__(self, system: System):

        self.system = system

        self.indices = system.indices
        self.constants = system.constants
        self.coordinates = system.coordinates
        self.variables = system.coordinates + system.variables + tuple(system.definitions.keys())

        self.input = cast(tuple[Function, ...], system.state_space())

        unknowns_: set[Function] = set()

        for f, definition in system.definitions.items():
            unknowns_.update({f})
            unknowns_.update({v for v in self.system.dependencies_of(definition) 
                              if not system.is_constant(v)})
        for eq in system.equations.values():
            unknowns_.update({v for v in self.system.dependencies_of(eq) 
                              if not system.is_constant(v)})
        for v in self.input:
            if v in unknowns_:
                unknowns_.remove(v)

        unknowns = set()
        for u in unknowns_:
            already_exixts = False
            for v in unknowns.copy():
                if v.is_general_form_of(u): 
                    already_exixts = True
                    break
                if u.is_general_form_of(v): 
                    unknowns.remove(v)
                    break
            if not already_exixts: 
                unknowns.add(u)

        unknowns = list(unknowns)

        print(f'Unknowns: {unknowns}')
        print(f'Variables: {self.variables}')

        indices: list[tuple[Index, ...]] = []
        equations: list[Expr] = []
        jacobian = []
        
        for f, definition in system.definitions.items():
            equation = f - definition #type:ignore
            equations.append(equation)
            indices.append(tuple(f.index.keys()))
            derivatives = {}
            for v in unknowns:
                derivative = sp.diff(equation, v)
                if derivative != 0:
                    derivatives[v] = derivative
            jacobian.append(derivatives)
        for eq in system.equations.values():
            equation = eq.lhs - eq.rhs #type:ignore
            equations.append(equation)
            indices.append(self.system.free_index_of(sp.Eq(eq.lhs, eq.rhs)))
            derivatives = {}
            for v in unknowns:
                derivative = sp.diff(equation, v)
                if derivative != 0:
                    derivatives[v] = derivative
            jacobian.append(derivatives)

        self.unknowns = tuple(unknowns)
        self.equations = tuple(equations)
        self.jacobian = jacobian

        # print(indices)
        print(f'Equations: {len(equations)}, unknowns: {len(unknowns)}')

        printer = PythonPrinter()

        args_str = ', '.join([python_name(v.name) for v in self.constants + self.indices + self.variables])
        constants_str = ', '.join([python_name(v.name) for v in self.constants])

        equations_str = f'def equations_generated({args_str}, _):\n'
        for i, eq in enumerate(equations):
            equations_str += f'  _[{i}] = {printer.doprint(eq)}\n'
        # equations_str += f'  return _\n'

        unknowns_indices = {v: i for i, v in enumerate(self.unknowns)}

        jacobian_rows = []
        jacobian_cols = []

        variables_set = set(self.variables)

        jacobian_str = f'def jacobian_generated({args_str}, _):\n'
        jacobian_initial_str = f'def jacobian_initial_generated({constants_str}, _):\n'
        n = 0
        for i, derivatives in enumerate(self.jacobian):
            for v, derivative in derivatives.items():
                j = unknowns_indices[v]
                # print(i, j, v, derivative, self.system.dependencies_of(derivative) & unknowns_set)
                # print(derivative.is_constant())
                if derivative == 0:
                    continue

                jacobian_rows.append(i)
                jacobian_cols.append(j)

                if self.system.dependencies_of(derivative) & variables_set:
                    # jacobian_str += f'  _[{i}, {j}] = {printer.doprint(derivative)}\n'
                    jacobian_str += f'  _[{n}] = {printer.doprint(derivative)}\n'
                else:
                    jacobian_initial_str += f'  _[{n}] = {printer.doprint(derivative)}\n'
                
                n += 1

        # jacobian_str += f'  return _\n'
        # equation_str = '[\n  ' + ',\n  '.join([cast(str, printer.doprint(eq)) for eq in equations]) + '\n]'

        setter_str = f'def setter_generated({args_str}, _):\n'
        for i, unknown in enumerate(self.unknowns):
            setter_str += f'  {printer.doprint(unknown)} = _[{i}]\n'

        # print(equations_str)
        # print(jacobian_str)
        # print(jacobian_initial_str)
        # print(setter_str)

        exec(equations_str, globals())
        exec(jacobian_str, globals())
        exec(jacobian_initial_str, globals())
        exec(setter_str, globals())

        self.equations_generated = equations_generated #type:ignore
        self.jacobian_generated = jacobian_generated #type:ignore
        self.jacobian_initial_generated = jacobian_initial_generated #type:ignore
        self.setter_generated = setter_generated #type:ignore

        self.jacobian_rows = jacobian_rows
        self.jacobian_cols = jacobian_cols

    def run(self, condition: dict[name_type, float], 
            newton_max_iter: int = 100, newton_tol: float = 1e-8,
            initial_epsilon: float = 1e-6) -> Result:
        condition_ = { self.system(name): value for name, value in condition.items()}

        index = self.indices[0]
        index_start = index.min or 0
        index_end = index.max or 0
        if index_start and index_start in condition_:
            index_start = condition_[index_start]
        if index_end and index_end in condition_:
            index_end = condition_[index_end]
        N = cast(int, index_end) - cast(int, index_start) + 1

        print(f'{index} = {index_start}, ..., {index_end}')

        value_constants = []
        for v in self.constants:
            if v in condition_:
                value_constants.append(condition_[v])
            else:
                raise ValueError(f'Value of {v} must be provided in condition')
            
        value_indices = []
        for v in self.indices:
            value_indices.append(np.arange(index_start, cast(int, index_end) + 1))

        value_variables = []
        for v in self.variables:
            if v in self.input:
                if v in condition_:
                    values = np.empty((N+1,), dtype=float)
                    values[:] = np.nan
                    values[0] = condition_[v]
                    value_variables.append(values)
                else:
                    raise ValueError(f'Value of {v} must be provided in condition')
            else:
                values = np.empty((N+1,), dtype=float)
                values[:] = np.nan
                values[0] = initial_epsilon * np.random.randn()  # Random initial guess
                value_variables.append(values)

        value_unknowns = np.zeros((len(self.unknowns)), dtype=float)
        value_equation = np.zeros((len(self.unknowns)), dtype=float)
        
        value_jacobian_data = np.zeros(len(self.jacobian_rows))
        self.jacobian_initial_generated(*value_constants, value_jacobian_data)

        value_jacobian = sparse.coo_matrix((value_jacobian_data, (self.jacobian_rows, self.jacobian_cols)), 
                                           shape=(len(self.unknowns), len(self.unknowns))).tocsr()
        
        # np.zeros((len(self.unknowns), len(self.unknowns)), dtype=float)

        result = Result(self.system)

        psudo_inverse_warning = False

        for i_ in tqdm(value_indices[0]):

            for values in value_variables:
                if np.isnan(values[i_ + 1]):
                    values[i_ + 1] = values[i_] # + initial_epsilon * np.random.randn()

            for newton_iter in range(0, newton_max_iter + 1):
                try:
                    self.equations_generated(*value_constants,i_,*value_variables, value_equation)
                    # self.jacobian_generated(*value_constants,i_,*value_variables, value_jacobian)
                    self.jacobian_generated(*value_constants,i_,*value_variables, value_jacobian_data)
                    value_jacobian.data = value_jacobian_data

                    # print('vars 0', np.array(value_variables)[:,newton_iter])
                    # print('vars 1', np.array(value_variables)[:,newton_iter + 1])
                    # print('equations', value_equation)


                    # with np.printoptions(threshold=np.inf):
                    #     print('jacobian',value_jacobian)

                    # print('jacobian', np.linalg.matrix_rank(value_jacobian), value_jacobian.shape)

                    with warnings.catch_warnings(record=True) as w:
                        # value_residual = linalg.solve(value_jacobian, value_equation)
                        warnings.simplefilter("always", category=sparse.linalg.MatrixRankWarning)
                        value_residual = sparse.linalg.spsolve(value_jacobian, value_equation)

                        if any(issubclass(warning.category, sparse.linalg.MatrixRankWarning) for warning in w):

                            value_jacobian_array = value_jacobian.toarray()
                            if not psudo_inverse_warning:
                                psudo_inverse_warning = True
                                # value_jacobian_array = value_jacobian.toarray()
                                print(f'warning: Singular matrix encountered in Newton method, using pseudo-inverse. rank: {np.linalg.matrix_rank(value_jacobian_array)} / {value_jacobian_array.shape[0]}')
                                
                                # u, s, vh = np.linalg.svd(value_jacobian_array)
                                # print('Singular values:', s)

                                import matplotlib.pyplot as plt
                                plt.imshow(value_jacobian.toarray(), cmap='viridis', interpolation='nearest')
                                plt.colorbar()
                                plt.show()
                            value_residual = linalg.pinv(value_jacobian_array) @ value_equation
                            # value_residual = sparse.linalg.lgmres(value_jacobian, value_equation)[0]
                        
                    value_unknowns -= value_residual

                    self.setter_generated(*value_constants,i_,*value_variables, value_unknowns)

                    if np.linalg.norm(value_residual) < newton_tol:
                        result.newton_converged_iters.append(newton_iter)
                        break
                    if newton_iter == newton_max_iter:
                        raise ValueError('Newton did not converge in {newton_max_iter} iterations')
                except:
                    raise Exception(f'Exception at index {i_}, iter {newton_iter}')

        result.set_data(self.constants, value_constants)
        result.set_data(self.variables, value_variables, N=N)

        return result
                

    def plot_jacobian(self, ticks = True):
        variables = set(self.variables)
        jacobian_matrix = np.zeros((len(self.equations), len(self.unknowns)), dtype=int)
        for i, row in enumerate(self.jacobian):
            for j, var in enumerate(self.unknowns):
                if var in row:
                    if self.system.dependencies_of(row[var]) & variables:
                        jacobian_matrix[i, j] = 2
                    else:
                        jacobian_matrix[i, j] = 1

        plt.figure(figsize=(5, 5))
        plt.imshow(jacobian_matrix, cmap='Greys', interpolation='none')
        plt.title("Jacobian")
        if ticks:
            plt.xticks(ticks=range(len(self.unknowns)), 
                       labels=[f'${self.system.latex(v)}$' for v in self.unknowns], rotation=90)
        plt.gca().xaxis.set_ticks_position('top')
        plt.gca().xaxis.set_label_position('top')
        plt.show()

        # print('shape:', jacobian_matrix.shape)
        # print('rank:', np.linalg.matrix_rank(jacobian_matrix))

    