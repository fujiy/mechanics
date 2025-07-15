from typing import Union, Any, Optional, Literal, ParamSpec, cast, overload
from typing_extensions import Self
import inspect
from collections import defaultdict
from pprint import pprint
import sympy as sp
from sympy.core.cache import cacheit
import sympy.core.function as spf
import sympy.core.containers as spc
from sympy.printing.latex import LatexPrinter
import matplotlib.pyplot as plt

from .util import *
from .symbol import BaseSpace, Index, Expr, Basic, Function, Equation, Space
from .space import R

class System:
    _builtins:    dict[str, Any]
    _dict:        dict[str, Any]

    _base_space:  list[BaseSpace]
    _indices:     list[Index]
    _functions:   list[Function]
    _coordinates: list[Function]
    _constants:   list[Function]
    _definitions: dict[str, Expr]
    _equations:   dict[str, Equation]

    _history:    list[Basic] = []

    def __init__(self, space: Optional[name_type] = None):
        

        self._dict = {}
        self._builtins = { k: getattr(sp, k) for k in dir(sp) if not k.startswith('_') }\
                       | { 'diff': self.diff, 'eval': self.eval, 'collect': self.collect,}

        self._base_space  = []
        self._indices     = []
        self._functions   = []
        self._coordinates = []
        self._constants   = []
        self._definitions = {}
        self._equations   = {}
        
        time_diff_notation = lambda s, n: f'\\{"d" * n}ot{{{s}}}'
        if space: self.add_space(space, diff_notation={space: time_diff_notation})

    # Basic declaration

    def add_space(self, name: name_type, 
                #   min: Optional[expr_type] = None, 
                #   max: Optional[expr_type] = None,
                  diff_notation = None) -> Self:
        symbols = make_symbol(name)
        # if bool(min) != bool(max): 
        #     raise ValueError('Both min and max must be provided')
        # if len(symbols) > 1 and min is not None: 
        #     raise ValueError('Only one space variable can be provided with min and max')
        
        for symbol in symbols: 
            # if min is None:
            #     min_ = self.__add_function(f'{symbol.name}_0', base_space=())[0]
            # else:
            #     min_ = self(min, return_as_tuple=False)
            # if max is None:
            #     max_ = self.__add_function(f'{symbol.name}_1', base_space=())[0]
            # else:
            #     max_ = self(max, return_as_tuple=False)

            if diff_notation is not None:
                if isinstance(diff_notation, dict):
                    diff_notation_ = diff_notation.get(symbol.name, None)
                else:
                    diff_notation_ = diff_notation
            else:
                diff_notation_ = None

            base_space = BaseSpace(symbol.name, diff_notation=diff_notation_)
            self._base_space.append(base_space)
            self.__register(symbol.name, base_space)

            if diff_notation_:
                for n in range(1, 5):
                    self._dict.update({ 
                        python_name(diff_notation_('', n)): lambda expr, n=n: self.diff(expr, base_space, n)})

        return self
    
    def add_index(self, name: name_type, min: expr_type, max: expr_type) -> Self:
        symbols = make_symbol(name)
        indices = []
        for symbol in symbols: 
            index = Index(symbol.name, self(min, manipulate=False, return_as_tuple=False), self(max, manipulate=False, return_as_tuple=False))
            indices.append(index)
            self._indices.append(index)
            self.__register(symbol.name, index)
        self.__add_history(to_single_or_tuple(indices))
        return self

    def add_coordinate(self, name: name_type, 
                       index: Union[name_type, tuple_ish[Index], None] = None,
                       base_space: Union[name_type, tuple_ish[BaseSpace], None] = None, 
                       space: expr_type = R, 
                       **options) -> Self:
        coordinates = self.__add_function(name, index=index, base_space=base_space, space=space, **options)
        self._coordinates.extend(coordinates)
        self.__add_history(to_single_or_tuple(coordinates))
        return self
    
    def add_variable(self, name: name_type, 
                     index: Union[name_type, tuple_ish[Index], None] = None,
                     base_space: Union[name_type, tuple_ish[BaseSpace], None] = None, 
                     space: expr_type = R, **options) -> Self:
        variables = self.__add_function(name, index=index, base_space=base_space, space=space, **options)
        self.__add_history(to_single_or_tuple(variables))
        return self
    
    def add_constant(self, name: name_type,
                     index: Union[name_type, tuple_ish[Index], None] = None,
                     **options) -> Self:
        constants = self.__add_function(name, index=index, base_space=(), **options)
        self._constants.extend(constants)
        self.__add_history(to_single_or_tuple(constants))
        return self

    def define(self, name: name_type, expr: expr_type, 
               condition: Union[expr_type, tuple[expr_type, expr_type]] = sp.true,
               index: Optional[expr_type] = None,
               **options) -> Self:
        symbols = make_symbol(name, **options)
        exprs = cast(tuple[Expr, ...], self(expr, return_as_tuple=True, **options))
        if isinstance(condition, tuple):
            condition_ = sp.Eq(self(condition[0]), self(condition[1]))
        elif condition == sp.true:
            condition_ = sp.true
        else:
            condition_ = self(condition, **options)
        if len(symbols) != len(exprs): 
            raise ValueError(f'Number of names and exprs must be the same, {({len(symbols)})} vs {({len(exprs)})}')
        if index:
            index_ = self(index, return_as_tuple=True, manipulate=False)
            if any([i not in self._indices for i in index_]):
                raise ValueError(f'Index \'{index}\' is not exists')
            index_ = cast(tuple[Index], index_)
        else:
            index_ = tuple()

        funcs = []

        for symbol, expr in zip(symbols, exprs):
            base_space = self.base_space_of(expr)
            index__ = index_ + self.free_index_of(expr)

            if symbol.name in self._definitions:
                func = self[symbol.name]
                if not isinstance(func, Function):
                    raise ValueError(f'Name \'{symbol.name}\' is not a function')
                func = cast(Function, func)
                if tuple(func.index.keys()) != index__:
                    raise ValueError(f'Index of \'{symbol.name}\' is not the same, {func.index.keys()} vs {index__}')
                if tuple(func.base_space.keys()) != base_space:
                    raise ValueError(f'Base space of \'{symbol.name}\' is not the same')

                definition = self._definitions[symbol.name]
                if isinstance(definition, sp.Piecewise):
                    cases = cast(list[tuple[Expr, Basic]], list(definition.args))
                else:
                    cases = [(definition, sp.true)]

                if condition in [ case[1] for case in cases ]:
                    raise ValueError(f'Function definition already exists for given condition: {condition}')
                cases.append((expr, condition_))

                self._definitions[symbol.name] = sp.Piecewise(*cases)
                funcs.append(self._definitions[symbol.name])

            else:
                if symbol.name not in self._dict:
                    func = self.__add_function(symbol.name, index=index__, base_space=base_space, space=R, **options)[0]
                    funcs.append(func)
                    self.__register(symbol.name, func)
                self._definitions[symbol.name] = sp.Piecewise((expr, condition_))
                funcs.append(self._definitions[symbol.name])

        self.__add_history(to_single_or_tuple(funcs))
        return self
    
    def equate(self, expr: expr_type, rhs: expr_type = sp.S.Zero, label: str = '', **options) -> Self:
        if isinstance(expr, Equation):
            rhs = expr.rhs
            expr = expr.lhs
            
        expr = cast(tuple[Expr, ...], self(expr, return_as_tuple=True, **options))
        rhs  = cast(tuple[Expr, ...], self(rhs,  return_as_tuple=True, **options))
        if len(rhs) == 1: rhs = rhs * len(expr)
        if len(expr) != len(rhs): 
            raise ValueError(f'Number of lhs and rhs must be the same, {({len(expr)})} vs {({len(rhs)})}')

        if label: label_ = to_tuple(label)
        else:     label_ = [f'Eq_{len(self._equations)}']
        if len(label_) != len(expr): 
            if len(label_) == 1: 
                label_ = [label_[0] + f'_{i}' for i in range(len(expr))]
            else:               
                raise ValueError(f'Number of labels and exprs must be the same, {({len(label_)})} vs {({len(expr)})}')
        
        equations = {} 
        for expr, rhs, l in zip(expr, rhs, label_):
            if l in self._equations:
                raise ValueError(f'Equation \'{l}\' already exists')
            equation = cast(Equation, Equation(expr, rhs))
            equation._label = l
            self.__register(l, equation)
            equations[l] = equation
        self._equations.update(equations)

        return self
    
    def __register(self, name: name_type, expr: Any):
        name = python_name(name)
        if name in self._dict and expr != self._dict[name]: 
            raise ValueError(f'Name \'{name}\' already exists')
        self._dict[name] = expr

    def __add_function(self, name: name_type,
                        index: Union[name_type, tuple_ish[Index], None] = None,
                        base_space: Union[name_type, tuple_ish[BaseSpace], None] = None, 
                        space: Space = R, **options) \
        -> tuple[Function, ...]:
        options = { 'real': True } | options
        if base_space is None: base_space_ = tuple(self._base_space)
        else:                  base_space_ = cast(tuple[BaseSpace, ...], self(base_space, return_as_tuple=True))
        index_ = (index and cast(tuple[Index, ...], self(index, manipulate=False, return_as_tuple=True))) or tuple()
        symbols = make_symbol(name, **options)
        functions = tuple(Function.make(symbol.name, index=index_, base_space=base_space_, space=space) 
                          for symbol in symbols)
        for func in functions: self.__register(func.name, func) #type:ignore
        self._functions.extend(functions)
        return functions
    
    def solve(self, expr: expr_type, var: expr_type, label: tuple_ish[str] = '') -> Self:
        exprs = self(expr, evaluate=True, return_as_tuple=True, simplify=False, collect=False)
        var_ = self(var, return_as_tuple=True, simplify=False, collect=False)
        
        label = to_tuple(label)

        solutions = sp.solve(exprs, var_, dict=True)

        if not solutions:
            raise ValueError(f'No solutions {var_} found for {exprs}')

        for i, sol in enumerate(solutions):
            for n, (v, s) in enumerate(sol.items()):
                if label[n]: l = label[n]
                else:        l = self.latex(v)
                if len(solutions) > 1:
                    self.define(f'{{{l}}}_{i}', s)
                else:
                    self.define(l, s)

        return self
    
    # Get info

    def state_space(self, time: Optional[name_type] = None) -> tuple[Expr, ...]:
        if time is None:
            if not self._base_space:
                return tuple()
            time_ = self.base_space[0]
        else:
            time_ = cast(BaseSpace, self[time])
        
        X = []
        for q in self.coordinates:
            if time_ in q.base_space.keys():
                X.append(q)
        for q in self.coordinates:
            if time_ in q.base_space.keys():
                X.append(sp.diff(q, time_))

        return tuple(X)
    
    def base_space_of(self, expr: expr_type) -> tuple[BaseSpace, ...]:
        exprs = self(expr, return_as_tuple=True, manipulate=False)
        base_space = set().union(*[self.__base_space_of(expr) for expr in exprs])
        return tuple( s for s in self._base_space if s in base_space )

    def __base_space_of(self, expr: Basic) -> set[BaseSpace]:
        base_space = []
        if not hasattr(expr, 'atoms'):
            return set()
        for symbol in expr.atoms(sp.Symbol):
            if symbol in self._base_space:
                base_space.append(symbol)
        for f in expr.atoms(spf.AppliedUndef):
            if f in self._functions: 
                base_space.extend(f.base_space.keys())
        return set(base_space)
    
    def free_index_of(self, expr: expr_type) -> tuple[Index, ...]:
        exprs = self(expr, return_as_tuple=True, manipulate=False)
        free_index = set().union(*[self.__free_index_of(expr) for expr in exprs])
        return tuple( i for i in self._indices if i in free_index )
    
    def __free_index_of(self, expr: Basic) -> set[Index]:
        free_index = []
        if not hasattr(expr, 'atoms'):
            return set()
        for symbol in expr.atoms(sp.Symbol):
            if symbol in self._indices:
                free_index.append(symbol)
        for f in expr.atoms(spf.AppliedUndef):
            if f in self._functions: 
                free_index.extend(f.free_index)
        return set(free_index)
    
    # def index_cases_of(self, expr: expr_type) -> dict[Index, tuple[Basic, ...]]:
    #     exprs = self(expr, return_as_tuple=True, simplify=False)
    #     index_cases = defaultdict(list)
    #     for expr in exprs:
    #         for s, n in expr.args[1:]:
    
    def dependencies_of(self, expr: Union[Expr, Equation]) -> set[Function]:
        return { s for s in expr.atoms(spf.AppliedUndef)
                  if isinstance(s, Function) }
    
    def is_constant(self, expr: expr_type) -> bool:
        exprs = self(expr, return_as_tuple=True, manipulate=False)
        constants = set(self.constants)
        for expr in exprs: #type:ignore
            if self.dependencies_of(expr) - constants:
                return False
        return True

    # Access
    
    def __getattr__(self, name: str) -> Expr:
        if name in self._dict: return self._dict[name]
        else:                  raise AttributeError(f'\'{name}\' is not exists')
        
    def __getitem__(self, name: name_type) -> Basic:
        expr_: Union[Basic, tuple[Basic, ...]]
        if isinstance(name, str): 
            name = python_name(name)
            if name in self._dict: return self._dict[name]
            else:                  raise KeyError(f'\'{name}\' is not exists')
        else:
            return name

    def __contains__(self, name: name_type) -> bool:
        name = python_name(name)
        return name in self._dict
        
    @overload
    def __call__(self, expr: expr_type, 
                 *,
                 return_as_tuple: Literal[True],
                 sum_for: Optional[name_type] = None,
                 evaluate=False, evaluate_top_level=True,
                 manipulate: Optional[bool] = True,
                 **options
                 ) \
        -> tuple[Basic, ...]:
        pass

    @overload
    def __call__(self, expr: expr_type, 
                 *,
                 return_as_tuple: Literal[False],
                 sum_for: Optional[name_type] = None,
                 evaluate=False, evaluate_top_level=True,
                 manipulate: Optional[bool] = True,
                 **options
                 ) \
        -> Basic:
        pass

    @overload
    def __call__(self, expr: expr_type, 
                 *,
                 sum_for: Optional[name_type] = None,
                 evaluate=False, evaluate_top_level=True,
                 manipulate: Optional[bool] = True,
                 **options
                 ) \
        -> Basic:
        pass

    # @cacheit
    def __call__(self, expr: expr_type, 
                 *,
                 return_as_tuple: Optional[bool] = None,
                 sum_for: Optional[name_type] = None,
                 evaluate=False, evaluate_top_level=True,
                 manipulate: Optional[bool] = True,
                 **options,
                 ) \
        -> Union[Basic, tuple[Basic, ...]]:
        expr_: Union[Basic, tuple[Basic, ...]]
        if isinstance(expr, str): 
            expr_ = eval(expr, globals() | self._builtins, self._dict | self.__history_map())
        else:
            expr_ = expr
        if sum_for:
            index = cast(Index, self(sum_for, return_as_tuple=False))
            if index not in self._indices:
                raise ValueError(f'Index \'{index.name}\' is not exists')
            expr_ = sp.summation(expr_, (index, index.min, index.max))
        if evaluate_top_level and getattr(expr_, '_is_definition', False): 
            expr_ = expr_.single_expr #type:ignore
        if evaluate:              
            expr_ = self.eval(expr_) #type:ignore
        if isinstance(expr_, spc.Tuple): 
            expr_ = tuple(expr_)
        if isinstance(expr_, list):      
            expr_ = tuple(expr_)

        if manipulate is None:
            manipulate_options = { 'collect': False, 'simplify': False, 'expand': False } | options
            expr_ = self.manipulate(expr_, **manipulate_options) #type:ignore
        elif manipulate:
            expr_ = self.manipulate(expr_, **options) #type:ignore

        if return_as_tuple == True and not isinstance(expr_, tuple):
            expr_ = (expr_,)
        if return_as_tuple == False and len(to_tuple(expr_)) != 1: 
            raise ValueError(f'Expression must be a single expression: {expr_}')
        return expr_
    
    # Operations

    def eval(self, expr: expr_type) -> Union[Basic, tuple[Basic, ...]]:
        return self.__eval(expr)

    # @cacheit
    def __eval(self, expr: expr_type, recursive=set()) -> Union[Basic, tuple[Basic, ...]]:
        if isinstance(expr, str): expr_ = eval(expr, globals() | self._builtins, self._dict)
        else:                     expr_ = expr
        if isinstance(expr_, tuple) or isinstance(expr_, list): 
            return tuple(self.__eval(e) for e in expr_) #type:ignore
        evaluated = {}
        for f in expr_.atoms(spf.AppliedUndef):
            if f.name in self._definitions and f not in recursive:
                evaluated[f] = self.__eval(self._definitions[f.name].subs(f.args_subs), 
                                           recursive | set([f]))

        return cast(Basic, expr_.subs(evaluated))


    # Expression manipulation

    def manipulate(self, expr: Basic, collect=True, simplify=True, expand=False) -> Basic:
        if collect:
            expr = self.collect(expr)
        elif simplify:
            expr = sp.simplify(expr)
        if expand:
            expr = sp.expand(expr)
            
        return expr

    
    def collect(self, expr: expr_type, syms: Optional[expr_type] = None) -> Union[Basic, tuple[Basic, ...]]:
        expr_ = self(expr, manipulate=False)
        if syms:
            syms_ = self(syms, return_as_tuple=True, simplify=False, collect=False, evaluate=False)
        else:
            syms_ = None
        if isinstance(expr_, tuple):
            return tuple(self.collect(e, syms) for e in expr_) #type:ignore
        
        return self.__collect(expr_, syms_)

    def __collect(self, expr: Expr, syms: Optional[Function] = None) -> Expr:
        if syms:
            syms_ = self(syms, return_as_tuple=True, simplify=False, collect=False, evaluate=False)
        else:
            syms_ = reversed(self.state_space() + tuple(self.variables))

        return sum([sp.simplify(c) * v for v, c 
                    in sp.collect(sp.expand(expr), syms_, evaluate=False).items()]) #type:ignore
    

    @cacheit
    def diff(self, expr, *args, **kwargs) -> Expr:
        return sp.diff(self.eval(expr), 
                       *[self(a, evaluate_top_level=False) for a in args],
                       **kwargs)
    
    # History

    def __add_history(self, value: Any):
        self._history.append(value)

    def __history_map(self) -> dict[str, Any]:
        history_map = {}
        for n in range(1, 4):
            if len(self._history) <= n:
                return history_map
            history_map['_' * n] = self._history[-n]
        return history_map

    # Discretization

    def discretization(self) -> Any:
        from mechanics.discretization import DiscretizedSystem
        return DiscretizedSystem(self)

    # Solver
    def solver(self):
        from mechanics.solver import Solver
        return Solver(self)

    # Printing

    def latex(self, expr: expr_type, **options) -> str:
        expr_ = self(expr, manipulate=None, **options)
        return LatexPrinterModified(self).doprint(expr_)
    
    def show(self, expr: Optional[expr_type] = None, label: str = '', label_str: str = '', **options) -> Self:
        from IPython.display import display, Math #type:ignore
        options = {'manipulate': None} | options
        message = ''
        if label:
            message += label
        elif label_str:
            message += r'\mathrm{' + label_str + '}'
        if (label or label_str) and expr is not None:
            message += ':'
        if expr is not None:
            message += self.latex(self(expr, **options)) #type:ignore
        display(Math(message))
        return self

    def show_all(self) -> Self:
        from IPython.display import display, Math #type:ignore

        if len(self.coordinates) > 0:
            if len(self.coordinates) == 1:
                coords = self.latex(self.coordinates[0])
            else:
                coords = '(' + ', '.join([ self.latex(q) for q in self.coordinates ]) + ')'
            display(Math('Q = ' + self.configuration + r' \ni ' + coords))

        if self.constants:
            self.show(self.constants, label_str='Constants')

        if self.variables:
            self.show(self.variables, label_str='Variables')

        if self.definitions:
            self.show(label_str='Definitions:')
            for f, definition in self.definitions.items():
                self.show(sp.Eq(f, definition))

        if self.equations:
            self.show(label_str='Equations')
            for label, eq in self.equations.items():
                self.show(eq, label=label)

        return self
    
    # Properties

    @property
    def base_space(self) -> tuple[BaseSpace, ...]:
        return tuple(self._base_space)
    
    @property
    def indices(self) -> tuple[Index, ...]:
        return tuple(self._indices)
    
    @property
    def configuration(self) -> str:
        return r' \times '.join(
            [str(q.space) for q in self._coordinates])
    
    @property
    def coordinates(self) -> tuple[Function, ...]:
        return tuple(self._coordinates)
    
    @property
    def variables(self) -> tuple[Function, ...]:
        # return tuple(set(self._functions) - set(self._constants) - set(self._coordinates) - set(self._definitions.keys()))
        return tuple(v for v in self._functions
                     if v not in self._constants
                        and v not in self._coordinates
                        and v.name not in self._definitions)
    
    @property
    def constants(self) -> tuple[Function, ...]:
        return tuple(self._constants) 
    
    @property
    def definitions(self) -> dict[Function, Expr]:
        return { cast(Function, self(python_name(f), return_as_tuple=False)): definition 
                for f, definition in self._definitions.items() }
    
    @property
    def equations(self) -> dict[str, Equation]:
        return self._equations
    

class LagrangeSystem(System):

    def __init__(self, space: Optional[name_type] = 't'):
        super().__init__(space)

        
    def euler_lagrange_equation(self, L: expr_type, time_var='t', label='EL') -> Self:
        L_ = self(L, evaluate=True, return_as_tuple=False)
        time_var = self(time_var, return_as_tuple=False)

        equations: list[Expr] = []

        for q in self.coordinates:
            for q_n in q.enumerate():
                dLdq = self.diff(L_, q_n)
                equation = dLdq

                for s in self.base_space:
                    dLddq = self.diff(L_, self.diff(q_n, s))
                    d_dLddq_ds = self.diff(dLddq, s)
                    equation -= d_dLddq_ds #type:ignore

                if equation != 0:
                    equations.append(equation)

        self.equate(tuple(equations), label=label)

        return self

    def legendre_transform(self, L: expr_type = 'L', H: str = 'H') -> 'HamiltonSystem':
        t = self.base_space[0]
        L = self(L, evaluate=True, return_as_tuple=False)
        qs = self.coordinates
        p_values = [self.diff(L, self.diff(q, t)) for q in qs]
        # display(qs, p_values)

        system = HamiltonSystem(t)

        ps = []
        
        for q in self.coordinates:
            system.add_coordinate(q.name, 
                                  index=tuple(q.index.keys()), 
                                  base_space=tuple(q.base_space.keys()), 
                                  space=q.space)
            system.add_variable(f'p_{{{q.name}}}', 
                                index=tuple(q.index.keys()), 
                                base_space=tuple(q.base_space.keys()))
            ps.append(system[f'p_{{{q.name}}}'])

        # display([p_value - p for p, p_value in zip(ps, p_values)])
        # display([self.diff(q, t) for q in qs])

        p_subs = sp.solve([p_value - p for p, p_value in zip(ps, p_values)], 
                          [self.diff(q, t) for q in qs], dict=True)[0]
        
        # display(p_subs)

        # p_subs = zip(ps, p_values)

        for q in self.constants:
            system.add_constant(q.name, 
                                index=tuple(q.index.keys()), 
                                space=q.space)
            
        for q in self.variables:
            system.add_variable(q.name, 
                                index=tuple(q.index.keys()), 
                                base_space=tuple(q.base_space.keys()), 
                                space=q.space)
            
        for f, expr in self.definitions.items():
            if f.name in system: continue
            system.define(f.name, expr.subs(p_subs))

        system.define(H, (sum([p * self.diff(q, t) for p, q in zip(ps, qs)]) - L).subs(p_subs)) #type:ignore

        return system
    
class HamiltonSystem(System):

    pass

class LatexPrinterModified(LatexPrinter):

    def __init__(self, system) -> None:
        super().__init__()
        self._system = system

    def _print_Derivative(self, expr: Expr) -> str:

        notations = []
        no_notations = []

        for s, n in expr.args[1:]: #type:ignore
            if isinstance(s, BaseSpace) \
                and s in self._system._base_space \
                and s.diff_notation is not None:
                notations.append((s.diff_notation, n))
            else:
                no_notations.append((s, n))

        printed = self.doprint(expr.args[0])
        for notation, n in notations:
            printed = notation(printed, n)

        if no_notations:
            return super()._print_Derivative(sp.Derivative(sp.Symbol(printed), no_notations)) #type:ignore
        else:
            return printed
