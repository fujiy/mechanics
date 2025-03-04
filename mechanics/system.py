from typing import Union, Any, Optional, cast
import sympy as sp
from sympy.core.cache import cacheit
import sympy.core.function as spf
import sympy.core.containers as spc
from sympy.printing.latex import LatexPrinter
from mechanics.symbol import name_type, expr_type, make_symbol, python_name, to_tuple
from mechanics.variable import BaseSpace, Index, Variable, Function, Equation
from mechanics.space import R

class System:
    _base_space:  dict[str, BaseSpace]
    _indices:     dict[str, Index]
    _coordinates: dict[str, Variable]
    _variables:   dict[str, Variable]
    _functions:   dict[str, Function]
    _equations:   dict[str, Equation]
    _dict:        dict[str, Any]
    _builtins:    dict[str, Any]

    def __init__(self, space: Optional[name_type] = 't'):

        self._base_space  = {}
        self._indices     = {}
        self._coordinates = {}
        self._variables   = {}
        self._functions   = {}
        self._equations   = {}
        self._dict = {}

        self._builtins = { k: getattr(sp, k) for k in dir(sp) if not k.startswith('_') }\
                       | { 'diff': self.diff }

        if space: self.add_space(space)

    # Basic declaration

    def add_space(self, name: name_type, 
                  min: Optional[expr_type] = None, 
                  max: Optional[expr_type] = None) -> 'System':
        symbols = make_symbol(name)
        if bool(min) != bool(max): 
            raise ValueError('Both min and max must be provided')
        if len(symbols) > 1 and min is not None: 
            raise ValueError('Only one space variable can be provided with min and max')
        
        for symbol in symbols: 
            if min is None:
                min_ = self.__add_variable(f'{symbol.name}_0', base_space=())[0]
            else:
                min_ = self(min, must_be_single=True)
            if max is None:
                max_ = self.__add_variable(f'{symbol.name}_1', base_space=())[0]
            else:
                max_ = self(max, must_be_single=True)
            space = BaseSpace(symbol.name, min_, max_)
            self._base_space.update({ symbol.name: space })
            self.__register(symbol.name, space)

        return self
    
    def add_index(self, name: name_type, min: expr_type, max: expr_type) -> 'System':
        symbols = make_symbol(name)
        for symbol in symbols: 
            index = Index(symbol.name, self(min, must_be_single=True), self(max, must_be_single=True))
            self._indices.update({ symbol.name: index })
            self.__register(symbol.name, index)
        return self

    def add_coordinate(self, name: name_type, 
                       index: Optional[name_type] = None,
                       base_space: Optional[name_type] = None, 
                       space: expr_type = R, 
                       **options) -> 'System':
        coordinates = self.__add_variable(name, index=index, base_space=base_space, space=space, **options)
        self._coordinates.update({ q.name: q for q in coordinates})
        return self
    
    def add_variable(self, name: name_type, 
                     index: Optional[name_type] = None,
                     base_space: Optional[name_type] = None, 
                     space: expr_type = R, **options) -> 'System':
        self.__add_variable(name, index=index, base_space=base_space, space=space, **options)
        return self
    
    def add_constant(self, name: name_type,
                     index: Optional[name_type] = None,
                     **options) -> 'System':
        self.__add_variable(name, index=index, base_space=(), **options)
        return self

    def define(self, name: name_type, expr: expr_type, 
               condition: Union[expr_type, tuple[expr_type, expr_type]] = sp.true,
               **options) -> 'System':
        symbols = make_symbol(name, **options)
        exprs = self(expr, always_tuple=True, **options)
        if isinstance(condition, tuple):
            condition_ = sp.Eq(self(condition[0]), self(condition[1]))
        else:
            condition_ = self(condition, **options)
        if len(symbols) != len(exprs): 
            raise ValueError(f'Number of names and exprs must be the same, {({len(symbols)})} vs {({len(exprs)})}')

        for symbol, expr in zip(symbols, exprs):
            base_space = self.base_space_of(expr)
            free_index = self.free_index_of(expr)
            if symbol.name in self._dict:
                self._functions[symbol.name].add_definition(definition=cast(sp.Expr, expr), condition=condition_)
            else:
                func = Function.make(symbol.name, index=free_index, base_space=base_space,
                                     definition=cast(sp.Expr, expr), condition=condition_, **options)
                self.__register(symbol.name, func)
                self._functions.update({symbol.name: func})

        return self
    
    def equate(self, expr: expr_type, rhs: expr_type = sp.S.Zero, label: str = '') -> 'System':

        expr = cast(tuple[sp.Expr, ...], self(expr, always_tuple=True))
        rhs  = cast(tuple[sp.Expr, ...], self(rhs, always_tuple=True))
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

    def __add_variable(self, name: name_type,
                        index: Optional[name_type] = None,
                        base_space: Optional[expr_type] = None, 
                        space: expr_type = R, **options) \
        -> tuple[Variable, ...]:
        options = { 'real': True } | options
        if base_space is None: base_space_ = tuple(self._base_space.values())
        else:                  base_space_ = self(base_space, always_tuple=True)
        index_ = (index and self(index, always_tuple=True)) or ()
        symbols = make_symbol(name, **options)
        variables = tuple(Variable.make(symbol.name, index=index_, base_space=base_space_) for symbol in symbols)
        for var in variables: self.__register(var.name, var) #type:ignore
        self._variables.update({ var.name: var for var in variables })
        return variables
    
    # Advanced declaration
        
    def euler_lagrange_equation(self, L: expr_type, time_var='t', label='EL') -> 'System':
        L = self(L, must_be_single=True)
        time_var = self(time_var, must_be_single=True)

        equations: list[sp.Expr] = []

        for q in self.coordinates:
            dLdq = self.diff(L, q)
            equation = dLdq
            for s in self.base_space:
                dLddq = self.diff(L, self.diff(q, s))
                d_dLddq_ds = self.diff(dLddq, s)
                equation -= d_dLddq_ds

            equations.append(equation)

        self.equate(tuple(equations), label=label)

        return self
    
    # Get info
    
    def base_space_of(self, expr: expr_type) -> tuple[BaseSpace, ...]:
        exprs = self(expr, always_tuple=True)
        base_space = set().union(*[self.__base_space_of(expr) for expr in exprs])
        return tuple( s for s in self._base_space.values() if s in base_space )

    def __base_space_of(self, expr: sp.Basic) -> set[BaseSpace]:
        base_space = []
        for symbol in expr.atoms(sp.Symbol):
            if symbol.name in self._base_space:
                base_space.append(self._base_space[symbol.name])
        for f in expr.atoms(spf.AppliedUndef):
            name = getattr(f, 'name', None)
            if name in self._variables: 
                base_space.extend(self._variables[name].base_space)
            elif name in self._functions:
                base_space.extend(self._functions[name].base_space)
        return set(base_space)
    
    def free_index_of(self, expr: expr_type) -> tuple[Index, ...]:
        exprs = self(expr, always_tuple=True)
        free_index = set().union(*[self.__free_index_of(expr) for expr in exprs])
        return tuple( i for i in self._indices.values() if i in free_index )
    
    def __free_index_of(self, expr: sp.Basic) -> set[Index]:
        free_index = []
        for symbol in expr.atoms(sp.Symbol):
            if symbol.name in self._indices:
                free_index.append(self._indices[symbol.name])
        for f in expr.atoms(spf.AppliedUndef):
            name = getattr(f, 'name', None)
            if name in self._variables: 
                free_index.extend(self._variables[name].free_index)
            elif name in self._functions:
                free_index.extend(self._functions[name].free_index)
        return set(free_index)

    # Access
    
    def __getattr__(self, name: str) -> sp.Expr:
        if name in self._dict: return self._dict[name]
        else:                  raise AttributeError(f'\'{name}\' is not exists')
        
    def __getitem__(self, name: str) -> sp.Expr:
        name = python_name(name)
        if name in self._dict: return self._dict[name]
        else:                  raise KeyError(f'\'{name}\' is not exists')
        
    # @cacheit
    def __call__(self, expr: expr_type, evaluate=False, simplify=True, 
                 always_tuple=False, must_be_single=False, eval_top_level=True) \
        -> Union[sp.Basic, tuple[sp.Basic, ...]]:
        expr_: Union[sp.Basic, tuple[sp.Basic, ...]]
        if isinstance(expr, str): 
            expr_ = eval(expr, globals() | self._builtins, self._dict)
        else:
            expr_ = expr
        if eval_top_level and getattr(expr_, '_is_function', False): 
            expr_ = expr_.definition #type:ignore
        if evaluate:              
            expr_ = self.eval(expr_)
        if isinstance(expr_, spc.Tuple): 
            expr_ = tuple(expr_)
        if isinstance(expr_, list):      
            expr_ = tuple(expr_)
        if simplify: 
            if isinstance(expr_, tuple): 
                expr_ = tuple(sp.simplify(e) for e in expr_)
            else:
                expr_ = sp.simplify(expr_)
        if always_tuple and not isinstance(expr_, tuple):
            expr_ = (expr_,)
        if must_be_single and len(to_tuple(expr_)) != 1: 
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
            if f.name in self._functions and f not in recursive:
                evaluated[f] = self.__eval(f.definition, recursive | set([f]))

        return cast(sp.Basic, expr_.subs(evaluated))
    

    # @cacheit
    def diff(self, expr, *args, **kwargs) -> sp.Expr:
        return sp.diff(self.eval(expr), 
                       *[self(a, eval_top_level=False) for a in args],
                       **kwargs)
    

    # Discritization

    def discretize(self, index: name_type, index_min: expr_type, index_max: expr_type, step: expr_type,
                   space: Optional[expr_type] = None,
                   diff = [], integral=None) -> 'System':
        
        if space is None: space = tuple(self._base_space.values())[0]
        
        system = System(space=None)
        for s in self._base_space.values():
            if s != space:
                system.add_space(s.name, s.min, s.max)
        for c in self.constants: 
            system.add_constant(c.name)
        system.add_index(index, index_min, index_max)

        def replace_diff(expr: sp.Expr, *args):

            print('diff', expr, args, isinstance(expr, Variable), isinstance(expr, spf.AppliedUndef))
            print(expr._base_space, self.latex(sp.Derivative(expr, *args)))
            return expr

        print([ f.definition.replace(sp.Derivative, replace_diff) for f in self.functions ])

        return system
    

    # Printing

    def latex(self, expr: expr_type, **options) -> str:
        expr_ = self(expr, **options)
        return LatexPrinterModified().doprint(expr_)

    
    # Properties

    @property
    def base_space(self) -> tuple[sp.Symbol, ...]:
        return tuple(self._base_space.values())
    
    @property
    def indices(self) -> tuple[Index, ...]:
        return tuple(self._indices.values())
    
    @property
    def configuration(self) -> str:
        return r' \times '.join(
            [str(q.base_space) for q in self._coordinates.values()])
    
    @property
    def coordinates(self) -> tuple[Variable, ...]:
        return tuple(self._coordinates.values())
    
    @property
    def variables(self) -> tuple[Variable, ...]:
        return tuple(self._variables.values())
    
    @property
    def constants(self) -> tuple[Variable, ...]:
        return tuple(v for v in self._variables.values() if v.base_space == ())
    
    @property
    def functions(self) -> tuple[Function, ...]:
        return tuple(self._functions.values())
    
    @property
    def equations(self) -> tuple[Equation, ...]:
        return tuple(self._equations.values())

class LatexPrinterModified(LatexPrinter):

    def _print_Derivative(self, expr: sp.Expr) -> str:
        return super()._print_Derivative(expr) #type:ignore