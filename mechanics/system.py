from typing import Union, Any, Optional, cast
import sympy as sp
from sympy.core.cache import cacheit
import sympy.core.function as spf
import sympy.core.containers as spc
from mechanics.symbol import name_type, expr_type, make_symbol, python_name, to_tuple
from mechanics.variable import Variable, Function, Equation
from mechanics.space import R

class System:
    _base_space:  tuple[sp.Symbol, ...]
    _coordinates: dict[str, Variable]
    _variables:   dict[str, Variable]
    _functions:   dict[str, Function]
    _equations:   dict[str, Equation]
    _dict:        dict[str, Any]
    _builtins:    dict[str, Any]

    def __init__(self, base_space: name_type = 't'):

        self._coordinates = {}
        self._variables   = {}
        self._functions   = {}
        self._equations   = {}
        self._dict = {}

        self._base_space = make_symbol(base_space, real=True)
        for symbol in self._base_space: self.__register(symbol.name, symbol)

        self._builtins = { k: getattr(sp, k) for k in dir(sp) if not k.startswith('_') }\
                       | { 'diff': self.diff }

    # Declaration

    def add_coordinate(self, name: name_type, 
                       base_space: Optional[name_type] = None, 
                       space: expr_type = R, **options) -> 'System':
        coordinates = self.__add_variable(name, base_space, space=space, **options)
        self._coordinates.update({ q.name: q for q in coordinates})
        return self
    
    def add_variable(self, name: name_type, 
                     base_space: Optional[name_type] = None, 
                     space: expr_type = R, **options) -> 'System':
        self.__add_variable(name, base_space, space=space, **options)
        return self
    
    def add_constant(self, name: name_type, **options) -> 'System':
        self.__add_variable(name, base_space='', **options)
        return self

    def define(self, name: name_type, expr: expr_type, **options) -> 'System':
        symbols = make_symbol(name, **options)
        exprs = self(expr, always_tuple=True, **options)
        if len(symbols) != len(exprs): 
            raise ValueError(f'Number of names and exprs must be the same, {({len(symbols)})} vs {({len(exprs)})}')
        functions = {}
        for symbol, expr in zip(symbols, exprs):
            base_space = self.base_space_of(expr)
            func = spf.UndefinedFunction(symbol.name, bases=(Function,), **options)\
                                        (*base_space, expr=expr, **options) #type:ignore
            self.__register(symbol.name, func)
            functions.update({symbol.name: func})
        self._functions.update(functions)
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
    
    def base_space_of(self, expr: expr_type) -> tuple[sp.Symbol, ...]:
        exprs = self(expr, always_tuple=True)
        base_space = set.union(*[self.__base_space_of(expr) for expr in exprs])
        return tuple( s for s in self._base_space if s in base_space )

    # Access
    
    def __getattr__(self, name: str) -> sp.Expr:
        if name in self._dict: return self._dict[name]
        else:                  raise AttributeError(f'\'{name}\' is not exists')
        
    def __getitem__(self, name: str) -> sp.Expr:
        name = python_name(name)
        if name in self._dict: return self._dict[name]
        else:                  raise KeyError(f'\'{name}\' is not exists')
        
    @cacheit
    def __call__(self, expr: expr_type, evaluate=False, simplify=True, 
                 always_tuple=False, must_be_single=False, eval_top_level=True) \
        -> Union[sp.Basic, tuple[sp.Basic, ...]]:
        expr_: Union[sp.Basic, tuple[sp.Basic, ...]]
        if isinstance(expr, str): 
            expr_ = eval(expr, globals() | self._builtins, self._dict)
        else:
            expr_ = expr
        if eval_top_level and getattr(expr, '_is_function', False): 
            expr_ = expr.expr #type:ignore
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

    @cacheit
    def eval(self, expr: expr_type) -> Union[sp.Basic, tuple[sp.Basic, ...]]:
        if isinstance(expr, str): expr_ = eval(expr, globals() | self._builtins, self._dict)
        else:                     expr_ = expr
        if isinstance(expr_, tuple): 
            return tuple(self.eval(e) for e in expr_)
        evaluated = {}
        for f in expr_.atoms(spf.AppliedUndef):
            if f.name in self._functions:
                evaluated[f] = self.eval(self._functions[f.name].expr)

        return cast(sp.Basic, expr_.subs(evaluated))

    @cacheit
    def diff(self, expr, *args, **kwargs) -> sp.Expr:
        return sp.diff(self.eval(expr), 
                       *[self(a, eval_top_level=False) for a in args],
                       **kwargs)
    

    # Discritization

    def discretize(self, space: name_type, step: name_type, index: name_type, index_min: name_type, index_max) -> 'System':

        return self
    

    
    # Properties

    @property
    def base_space(self) -> tuple[sp.Symbol, ...]:
        return self._base_space
    
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
    
    
    # Internal

    def __register(self, name: name_type, expr: Any):
        name = python_name(name)
        if name in self._dict and expr != self._dict[name]: 
            raise ValueError(f'Name \'{name}\' already exists')
        self._dict[name] = expr


    def __add_variable(self, name: name_type,
                        base_space: Optional[name_type] = None, 
                        space: expr_type = R, **options) \
        -> tuple[Variable, ...]:
        options = { 'real': True } | options
        if base_space is None: base_space_ = self._base_space
        else:                  base_space_ = make_symbol(base_space, real=True)
        symbols = make_symbol(name, **options)
        variables = cast(tuple[Variable, ...],
                         tuple(spf.UndefinedFunction(symbol.name, bases=(Variable,), **options)
                               (*base_space_, **options)
                         for symbol in symbols))
        for var in variables: self.__register(var.name, var) #type:ignore
        self._variables.update({ var.name: var for var in variables })
        return variables

    def __base_space_of(self, expr: sp.Basic) -> set[sp.Expr]:
        if isinstance(expr, spf.AppliedUndef):
            name = getattr(expr, 'name', None)
            if name in self._variables: 
                return set(self._variables[name].base_space)
            elif name in self._functions:
                return set(self._functions[name].base_space)
        return set().union(*[self.__base_space_of(arg) for arg in expr.args])
