from typing import Union, Any, Optional, Literal, cast, overload
from collections import defaultdict
from pprint import pprint
import sympy as sp
from sympy.core.cache import cacheit
import sympy.core.function as spf
import sympy.core.containers as spc
from sympy.printing.latex import LatexPrinter
import matplotlib.pyplot as plt

from .util import name_type, expr_type, make_symbol, python_name, to_tuple, tuple_ish
from .symbol import BaseSpace, Index, Expr, Function, Equation, Space
from .space import R

class System:
    _builtins:    dict[str, Any]
    _dict:        dict[str, Any]

    _base_space:  list[BaseSpace]
    _indices:     list[Index]
    _coordinates: list[Function]
    _functions:   list[Function]
    _definitions: dict[str, Expr]
    _equations:   dict[str, Equation]

    def __init__(self, space: Optional[name_type] = 't'):
        

        self._dict = {}
        self._builtins = { k: getattr(sp, k) for k in dir(sp) if not k.startswith('_') }\
                       | { 'diff': self.diff }

        self._base_space  = []
        self._indices     = []
        self._coordinates = []
        self._functions   = []
        self._definitions = {}
        self._equations   = {}
        
        time_diff_notation = lambda s, n: f'\\{"d" * n}ot{{{s}}}'
        if space: self.add_space(space, diff_notation={'t': time_diff_notation})

    # Basic declaration

    def add_space(self, name: name_type, 
                #   min: Optional[expr_type] = None, 
                #   max: Optional[expr_type] = None,
                  diff_notation = None) -> 'System':
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
    
    def add_index(self, name: name_type, min: expr_type, max: expr_type) -> 'System':
        symbols = make_symbol(name)
        for symbol in symbols: 
            index = Index(symbol.name, self(min, return_as_tuple=False), self(max, return_as_tuple=False))
            self._indices.append(index)
            self.__register(symbol.name, index)
        return self

    def add_coordinate(self, name: name_type, 
                       index: Union[name_type, tuple_ish[Index], None] = None,
                       base_space: Union[name_type, tuple_ish[BaseSpace], None] = None, 
                       space: expr_type = R, 
                       **options) -> 'System':
        coordinates = self.__add_function(name, index=index, base_space=base_space, space=space, **options)
        self._coordinates.extend(coordinates)
        return self
    
    def add_variable(self, name: name_type, 
                     index: Union[name_type, tuple_ish[Index], None] = None,
                     base_space: Union[name_type, tuple_ish[BaseSpace], None] = None, 
                     space: expr_type = R, **options) -> 'System':
        self.__add_function(name, index=index, base_space=base_space, space=space, **options)
        return self
    
    def add_constant(self, name: name_type,
                     index: Union[name_type, tuple_ish[Index], None] = None,
                     **options) -> 'System':
        self.__add_function(name, index=index, base_space=(), **options)
        return self

    def define(self, name: name_type, expr: expr_type, 
               condition: Union[expr_type, tuple[expr_type, expr_type]] = sp.true,
               index: Optional[expr_type] = None,
               **options) -> 'System':
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
            index_ = self(index, return_as_tuple=True)
            if any([i not in self._indices for i in index_]):
                raise ValueError(f'Index \'{index}\' is not exists')
            index_ = cast(tuple[Index], index_)
        else:
            index_ = tuple()

        for symbol, expr in zip(symbols, exprs):
            base_space = self.base_space_of(expr)
            index__ = index_ + self.free_index_of(expr)

            if symbol.name in self.definitions:
                func = self[symbol.name]
                if not isinstance(func, Function):
                    raise ValueError(f'Name \'{symbol.name}\' is not a function')
                func = cast(Function, func)
                if func.index != index__:
                    raise ValueError(f'Index of \'{symbol.name}\' is not the same')
                if func.base_space != base_space:
                    raise ValueError(f'Base space of \'{symbol.name}\' is not the same')

                definition = self._definitions[symbol.name]
                if isinstance(definition, sp.Piecewise):
                    cases = cast(list[tuple[sp.Expr, sp.Basic]], definition.args)
                else:
                    cases = [(definition, sp.true)]

                if condition in [ case[1] for case in cases ]:
                    raise ValueError(f'Function definition already exists for given condition: {condition}')
                cases.append((expr, condition_))

                print(symbol.name, cases)

                self._definitions[symbol.name] = sp.Piecewise(*cases)

            else:
                func = self.__add_function(symbol.name, index=index__, base_space=base_space, space=R, **options)[0]
                self.__register(symbol.name, func)
                self._definitions[symbol.name] = expr

        return self
    
    def equate(self, expr: expr_type, rhs: expr_type = sp.S.Zero, label: str = '') -> 'System':

        expr = cast(tuple[Expr, ...], self(expr, return_as_tuple=True))
        rhs  = cast(tuple[Expr, ...], self(rhs, return_as_tuple=True))
        if len(rhs) == 1: rhs = rhs * len(expr)
        if len(expr) != len(rhs): 
            raise ValueError(f'Number of lhs and rhs must be the same, {({len(expr)})} vs {({len(rhs)})}')

        if label: label_ = to_tuple(label)
        else:     label_ = [f'eq{len(self._equations)}']
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
        index_ = (index and cast(tuple[Index, ...], self(index, return_as_tuple=True))) or tuple()
        symbols = make_symbol(name, **options)
        functions = tuple(Function.make(symbol.name, index=index_, base_space=base_space_, space=space) 
                          for symbol in symbols)
        for func in functions: self.__register(func.name, func) #type:ignore
        self._functions.extend(functions)
        return functions
    
    # Advanced declaration
        
    def euler_lagrange_equation(self, L: expr_type, time_var='t', label='EL') -> 'System':
        L_ = self(L, evaluate=True, return_as_tuple=False)
        time_var = self(time_var, return_as_tuple=False)

        equations: list[Expr] = []

        for q in self.coordinates:
            dLdq = self.diff(L_, q)
            equation = dLdq

            for s in self.base_space:
                dLddq = self.diff(L_, self.diff(q, s))
                d_dLddq_ds = self.diff(dLddq, s)
                equation -= d_dLddq_ds #type:ignore

            equations.append(equation)



        self.equate(tuple(equations), label=label)

        return self
    
    # Get info

    def state_space(self, time: Optional[name_type] = None) -> tuple[Expr, ...]:
        if time is None:
            time_ = self.base_space[0]
        else:
            time_ = cast(BaseSpace, self(time, return_as_tuple=False))
        
        X = []
        for q in self.coordinates:
            if time_ in q.base_space:
                X.append(q)
        for q in self.coordinates:
            if time_ in q.base_space:
                X.append(sp.diff(q, time_))

        return tuple(X)
    
    def base_space_of(self, expr: expr_type) -> tuple[BaseSpace, ...]:
        exprs = self(expr, return_as_tuple=True, simplify=False)
        base_space = set().union(*[self.__base_space_of(expr) for expr in exprs])
        return tuple( s for s in self._base_space if s in base_space )

    def __base_space_of(self, expr: sp.Basic) -> set[BaseSpace]:
        base_space = []
        for symbol in expr.atoms(sp.Symbol):
            if symbol in self._base_space:
                base_space.append(symbol)
        for f in expr.atoms(spf.AppliedUndef):
            if f in self._functions: 
                base_space.extend(f.base_space)
        return set(base_space)
    
    def free_index_of(self, expr: expr_type) -> tuple[Index, ...]:
        exprs = self(expr, return_as_tuple=True, simplify=False)
        free_index = set().union(*[self.__free_index_of(expr) for expr in exprs])
        return tuple( i for i in self._indices if i in free_index )
    
    def __free_index_of(self, expr: sp.Basic) -> set[Index]:
        free_index = []
        for symbol in expr.atoms(sp.Symbol):
            if symbol in self._indices:
                free_index.append(symbol)
        for f in expr.atoms(spf.AppliedUndef):
            if f in self._functions: 
                free_index.extend(f.free_index)
        return set(free_index)
    
    def dependencies_of(self, expr: Union[Expr, Equation]) -> set[Function]:
        return { s for s in expr.atoms(spf.AppliedUndef)
                  if isinstance(s, Function) }
    
    def is_constant(self, expr: expr_type) -> bool:
        exprs = self(expr, return_as_tuple=True, simplify=False)
        result = True
        for expr in exprs: #type:ignore
            if isinstance(expr, Function):
                if expr.base_space or\
                    (expr.index and [i for i in expr.index if i in self._indices]):
                    result = False
                    break
            else:
                result = False
                break
        return result

    # Access
    
    def __getattr__(self, name: str) -> Expr:
        if name in self._dict: return self._dict[name]
        else:                  raise AttributeError(f'\'{name}\' is not exists')
        
    def __getitem__(self, name: str) -> Expr:
        name = python_name(name)
        if name in self._dict: return self._dict[name]
        else:                  raise KeyError(f'\'{name}\' is not exists')

    def __contains__(self, name: name_type) -> bool:
        name = python_name(name)
        return name in self._dict
        
    @overload
    def __call__(self, expr: expr_type, 
                 *,
                 return_as_tuple: Literal[True],
                 sum_for: Optional[name_type] = None,
                 simplify=True, evaluate=False, evaluate_top_level=True,
                 ) \
        -> tuple[sp.Basic, ...]:
        pass

    @overload
    def __call__(self, expr: expr_type, 
                 *,
                 return_as_tuple: Literal[False],
                 sum_for: Optional[name_type] = None,
                 simplify=True, evaluate=False, evaluate_top_level=True,
                 ) \
        -> sp.Basic:
        pass

    @overload
    def __call__(self, expr: expr_type, 
                 *,
                 sum_for: Optional[name_type] = None,
                 simplify=True, evaluate=False, evaluate_top_level=True,
                 ) \
        -> sp.Basic:
        pass

    # @cacheit
    def __call__(self, expr: expr_type, 
                 *,
                 return_as_tuple: Optional[bool] = None,
                 sum_for: Optional[name_type] = None,
                 simplify=True, evaluate=False, evaluate_top_level=True,
                 ) \
        -> Union[sp.Basic, tuple[sp.Basic, ...]]:
        expr_: Union[sp.Basic, tuple[sp.Basic, ...]]
        if isinstance(expr, str): 
            expr_ = eval(expr, globals() | self._builtins, self._dict)
        else:
            expr_ = expr
        if sum_for:
            index = cast(Index, self(sum_for, return_as_tuple=False))
            if index.name not in self._indices:
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
        if simplify: 
            if isinstance(expr_, tuple): 
                expr_ = tuple(sp.simplify(e) for e in expr_)
            else:
                expr_ = sp.simplify(expr_)
        if return_as_tuple == True and not isinstance(expr_, tuple):
            expr_ = (expr_,)
        if return_as_tuple == False and len(to_tuple(expr_)) != 1: 
            raise ValueError(f'Expression must be a single expression: {expr_}')
        return expr_
    
    # Operations

    def eval(self, expr: expr_type) -> Union[sp.Basic, tuple[sp.Basic, ...]]:
        return self.__eval(expr)

    # @cacheit
    def __eval(self, expr: expr_type, recursive=set()) -> Union[sp.Basic, tuple[sp.Basic, ...]]:
        if isinstance(expr, str): expr_ = eval(expr, globals() | self._builtins, self._dict)
        else:                     expr_ = expr
        if isinstance(expr_, tuple): 
            return tuple(self.__eval(e) for e in expr_) #type:ignore
        evaluated = {}
        for f in expr_.atoms(spf.AppliedUndef):
            if f.name in self._definitions and f not in recursive:
                evaluated[f] = self.__eval(self._definitions[f.name], recursive | set([f]))

        return cast(sp.Basic, expr_.subs(evaluated))
    

    @cacheit
    def diff(self, expr, *args, **kwargs) -> Expr:
        return sp.diff(self.eval(expr), 
                       *[self(a, evaluate_top_level=False) for a in args],
                       **kwargs)
    

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
        expr_ = self(expr, evaluate_top_level=False, simplify=False, **options)
        return LatexPrinterModified(self).doprint(expr_)
    
    def show(self, expr: expr_type, **options):
        from IPython.display import display, Math
        expr_ = self(expr, **options)
        display(Math(self.latex(expr_)))

    def show_all(self) -> 'System':
        from IPython.display import display, Math #type:ignore

        if len(self.coordinates) == 1:
            coords = self.latex(self.coordinates[0])
        else:
            coords = '(' + ', '.join([ self.latex(q) for q in self.coordinates ]) + ')'
        display(Math('Q = ' + self.configuration + r' \ni ' + coords))

        if self.constants:
            display(Math(r'\mathrm{constants}: ' 
                         + ', '.join([ self.latex(c) for c in self.constants ])))

        if self.variables:
            display(Math(r'\mathrm{variables}: '
                            + ', '.join([ self.latex(v) for v in self.variables ])))

        if self.definitions:
            display(Math(r'\mathrm{definitions}:'))
            for f, definition in self.definitions.items():
                display(Math(f'{self.latex(f)} = {self.latex(definition)}'))

        if self.equations:
            display(Math(r'\mathrm{equations}:'))
            for label, eq in self.equations.items():
                display(Math(f'\mathrm{{{label}}}: {self.latex(eq.lhs)} = {self.latex(eq.rhs)}'))

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
        return tuple(v for v in self._functions
                     if not self.is_constant(v) 
                        and v not in self._coordinates
                        and v.name not in self._definitions)
    
    @property
    def constants(self) -> tuple[Function, ...]:
        return tuple(v for v in self._functions
                     if self.is_constant(v) and v not in self._coordinates)
    
    @property
    def definitions(self) -> dict[Function, Expr]:
        return { cast(Function, self(f, return_as_tuple=False)): definition 
                for f, definition in self._definitions.items() }
    
    @property
    def equations(self) -> dict[str, Equation]:
        return self._equations
    
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
